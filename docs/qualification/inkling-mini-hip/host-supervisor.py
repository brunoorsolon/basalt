"""Host-specific, explicitly authorized issue-8 GPU envelope; no kernel/install changes."""
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import threading
import time

ROOT = Path('/mnt/storage/ai/redrock-issue8')
REPAIR = Path('/mnt/storage/ai/redrock-499/runtime-repair/build-002/rocr/lib/libhsa-runtime64.so.1.18.0')
EXPECTED = '63fa08fa9461d6f5e652fa50d232116b317091dbf0bbe2204799dfd705b564ca'
GPU = Path('/sys/bus/pci/devices/0000:03:00.0')
GIB = 1024**3
FAULT = re.compile(r'GPU reset|GPU [Hh]ang|ring .*timeout|amdgpu.*\b(?:fault|timeout)\b|Out of memory|oom-kill|Killed process|BUG:|Oops:|Kernel panic|general protection fault', re.I)

def require(ok, message):
    if not ok:
        raise RuntimeError(message)

def sample():
    values = {line.split(':')[0]: int(line.split()[1])*1024 for line in Path('/proc/meminfo').read_text().splitlines()}
    return {'monotonic': time.monotonic(), 'vram_bytes': int((GPU/'mem_info_vram_used').read_text()), 'available_ram_bytes': values['MemAvailable']}

def problem(reading):
    if reading['vram_bytes'] >= 28*GIB:
        return '28 GiB whole-device VRAM ceiling reached'
    if reading['available_ram_bytes'] < 8*GIB:
        return '8 GiB available host RAM floor reached'
    return None

def holders():
    result = subprocess.run(['/usr/bin/fuser','/dev/kfd'],capture_output=True,text=True)
    require(result.returncode in (0,1), 'cannot inspect compute ownership')
    return [int(p) for p in result.stdout.split()]

def main(label, command):
    os.umask(0o077)
    destination = ROOT/'evidence'/label
    require(not destination.exists(), 'choose a fresh evidence label')
    destination.mkdir()
    record = {'kernel':os.uname().release,'command':command,'status':'preflight','selected_pci':'0000:03:00.0','selected_uuid':'GPU-26508202313fa2ac','limits':{'unit_seconds':900,'process_group_ram_bytes':32*GIB,'whole_device_vram_bytes':28*GIB,'host_available_ram_floor_bytes':8*GIB},'source_commit':subprocess.check_output(['git','-C','/home/bruno/git-repos/brunoorsolon/redrock','rev-parse','HEAD'],text=True).strip()}
    def save():
        (destination/'supervisor.json').write_text(json.dumps(record,indent=2)+'\n')
    save()
    lease = None
    monitor = None
    proc = None
    was_active = False
    comfy_stopped = False
    unsafe = []
    unit = 'redrock-issue8-'+label
    try:
        with REPAIR.open('rb') as f:
            require(hashlib.file_digest(f,'sha256').hexdigest()==EXPECTED,'ROCr repair identity changed')
        record['rocr']={'path':str(REPAIR),'sha256':EXPECTED}
        sys.path.insert(0,'/home/bruno/.local/share/matuto/releases/2.1.0/agents')
        from kopernik import kopernik_runtime
        lease = kopernik_runtime.runtime_lease(exclusive=True)
        was_active = subprocess.run(['systemctl','--user','is-active','--quiet','comfyui.service']).returncode==0
        if was_active:
            ready_until = time.monotonic()+30
            while True:
                try:
                    queue = json.load(__import__('urllib.request',fromlist=['urlopen']).urlopen('http://127.0.0.1:8188/queue',timeout=5))
                    break
                except OSError:
                    require(time.monotonic()<ready_until,'ComfyUI service is not ready for an idle-queue check')
                    time.sleep(.2)
            require(not queue['queue_running'] and not queue['queue_pending'],'ComfyUI is no longer idle')
            subprocess.run(['systemctl','--user','stop','comfyui.service'],check=True,timeout=25)
            comfy_stopped = True
        require(not holders(),'another compute owner holds KFD')
        desktop = subprocess.check_output(['pgrep','-x','Hyprland'],text=True).split()
        require(len(desktop)==1,'cannot establish desktop identity')
        desktop_stat = Path('/proc/'+desktop[0]+'/stat').read_text().rsplit(') ',1)[1].split()[19]
        def desktop_alive():
            try:
                return Path('/proc/'+desktop[0]+'/stat').read_text().rsplit(') ',1)[1].split()[19]==desktop_stat
            except FileNotFoundError:
                return False
        cursor_output = subprocess.check_output(['journalctl','-k','-n','0','--show-cursor','--no-pager'],text=True)
        cursor = next(s.removeprefix('-- cursor: ') for s in cursor_output.splitlines() if s.startswith('-- cursor: '))
        record['kernel_start_cursor'] = cursor
        kernel_file = (destination/'kernel.jsonl').open('w')
        monitor = subprocess.Popen(['journalctl','-k','-f','--after-cursor='+cursor,'--no-pager','-o','json'],stdout=subprocess.PIPE,stderr=(destination/'kernel-stderr.log').open('w'),text=True)
        def observe_kernel():
            for line in monitor.stdout:
                kernel_file.write(line);kernel_file.flush()
                try:
                    if FAULT.search(str(json.loads(line).get('MESSAGE',''))):
                        unsafe.append('new kernel safety event; inspect kernel.jsonl')
                except ValueError:
                    unsafe.append('kernel observer received malformed data')
        thread = threading.Thread(target=observe_kernel,daemon=True);thread.start()
        argv = ['systemd-run','--user','--unit='+unit,'--wait','--pipe','--collect','-p','MemoryMax=32G','-p','MemorySwapMax=0','-p','RuntimeMaxSec=900','-p','TimeoutStopSec=10','-p','KillMode=control-group','bwrap','--unshare-user','--uid','0','--gid','0','--unshare-pid','--unshare-net','--die-with-parent','--cap-drop','ALL','--ro-bind','/','/','--ro-bind',str(REPAIR),'/usr/lib64/libhsa-runtime64.so.1.18.0','--proc','/proc','--dev','/dev','--dir','/dev/dri','--dev-bind','/dev/dri/renderD128','/dev/dri/renderD128','--dev-bind','/dev/kfd','/dev/kfd','--bind',str(ROOT/'evidence'),str(ROOT/'evidence'),'--tmpfs','/tmp','--','/usr/bin/env','-i','PATH=/usr/bin:/bin','HOME=/tmp','LC_ALL=C','ROCR_VISIBLE_DEVICES=GPU-26508202313fa2ac','HIP_VISIBLE_DEVICES=0','CUDA_VISIBLE_DEVICES=','IK2_KV=f16','IK2_EXPERT_MMA=0',*command]
        record.update(status='running',envelope=argv,baseline=sample());save()
        require(problem(record['baseline']) is None,'initial resource envelope unavailable')
        started = time.monotonic();last = started;peak_ram = 0
        with (destination/'run.log').open('w') as log, (destination/'telemetry.jsonl').open('w') as telemetry:
            proc = subprocess.Popen(argv,stdout=log,stderr=subprocess.STDOUT)
            while proc.poll() is None:
                reading=sample();telemetry.write(json.dumps(reading)+'\n');telemetry.flush()
                if time.monotonic()-last>=2:
                    unsafe.append('resource monitor stalled')
                    raise RuntimeError(unsafe[-1])
                last=time.monotonic()
                issue = problem(reading)
                if issue or unsafe or not desktop_alive():
                    unsafe.append(issue or ('desktop changed' if not desktop_alive() else unsafe[0]))
                    raise RuntimeError(unsafe[-1])
                if monitor.poll() is not None:
                    unsafe.append('kernel observer exited')
                    raise RuntimeError(unsafe[-1])
                require(time.monotonic()-started<910,'whole-run watchdog expired')
                control = Path('/sys/fs/cgroup/user.slice/user-1000.slice/user@1000.service/app.slice/'+unit+'.service/memory.peak')
                try:
                    peak_ram = max(peak_ram,int(control.read_text()))
                except FileNotFoundError:
                    pass
                time.sleep(.2)
            record.update(exit_code=proc.returncode,elapsed_seconds=time.monotonic()-started,cgroup_memory_peak_sampled_bytes=peak_ram)
        require(proc.returncode==0,'payload failed; inspect run.log')
        require(not unsafe,'new kernel safety event')
        record['status']='passed'
    except BaseException as error:
        record.update(status='failed',failure=f'{type(error).__name__}: {error}')
        raise
    finally:
        subprocess.run(['systemctl','--user','stop',unit],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,timeout=20)
        if proc is not None:
            proc.wait(timeout=15)
        until=time.monotonic()+15
        baseline_vram = record.get('baseline',sample())['vram_bytes']
        while (holders() or sample()['vram_bytes'] > baseline_vram + 512*1024**2) and time.monotonic()<until:
            time.sleep(.2)
        released = not holders() and sample()['vram_bytes'] <= baseline_vram + 512*1024**2
        if not released and proc is not None:
            unsafe.append('owned compute resources did not return near baseline within 15 seconds')
        record.update(compute_released=released,safety_events=unsafe,final_sample=sample())
        if monitor is not None:
            monitor.terminate();monitor.wait(timeout=5)
            final_kernel = subprocess.check_output(['journalctl','-k','--after-cursor='+record['kernel_start_cursor'],'--no-pager','-o','json'],text=True,timeout=5)
            (destination/'kernel-final.jsonl').write_text(final_kernel)
            if any(FAULT.search(str(json.loads(line).get('MESSAGE',''))) for line in final_kernel.splitlines()):
                unsafe.append('kernel safety event during execution/cleanup')
        if unsafe:
            record.update(status='failed',safety_events=unsafe)
        if comfy_stopped and released and not unsafe:
            subprocess.run(['systemctl','--user','start','comfyui.service'],check=True,timeout=25)
            record['comfyui_restored']=True
        if lease is not None:
            os.close(lease)
        save()

if __name__=='__main__':
    if sys.argv[1:]==['--self-test']:
        assert problem({'vram_bytes':28*GIB-1,'available_ram_bytes':8*GIB}) is None
        assert problem({'vram_bytes':28*GIB,'available_ram_bytes':8*GIB})
        assert problem({'vram_bytes':0,'available_ram_bytes':8*GIB-1})
        assert FAULT.search('amdgpu: ring gfx timeout')
        assert not FAULT.search('amdgpu initialization complete')
        print('Resource/fault boundary checks passed; no GPU initialization')
    else:
        main(sys.argv[1],sys.argv[2:])

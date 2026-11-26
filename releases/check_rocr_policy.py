"""CPU regression for the actual SVM-policy loop extracted from pinned fmm.c."""
from pathlib import Path
import subprocess
import sys
import tempfile

SOURCE = Path(sys.argv[1]).resolve()


def policy_loop(source):
    marker = source.index('ret = init_svm_apertures(svm_base, svm_limit, svm_alignment,')
    start = source.index('for (i = 0 ; i < num_of_sysfs_nodes ; i++) {', marker)
    opening = source.index('{', start)
    depth = 1
    end = opening + 1
    while depth:
        depth += (source[end] == '{') - (source[end] == '}')
        end += 1
    return source[start:end]


PREFIX = r'''
#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#define IS_CANONICAL_ADDR(x) ((x) == 1)
#define VOID_PTRS_SUB(a,b) ((a)-(b))
#define KFD_IOC_CACHE_POLICY_COHERENT 1
#define KFD_IOC_CACHE_POLICY_NONCOHERENT 0
#define GFX_VERSION_GFX950 950
#define HSAKMT_STATUS_ERROR 1
#define pr_err(...) ((void)0)
struct { unsigned gpu_id; unsigned gpuvm_limit; } process_apertures[3];
struct { unsigned gpu_id; unsigned node_id; } gpu_mem[3];
struct { uintptr_t base, limit; } aperture = {4096,8191};
struct { int disable_cache; typeof(aperture) *dgpu_alt_aperture; } svm = {0,&aperture};
unsigned gpu_count, calls, ids[3], flags[3], gfx_calls[3], gfx_count, fail_id;
static int gpu_mem_find_by_gpu_id(unsigned id) {
    for(unsigned j=0;j<gpu_count;j++) if(gpu_mem[j].gpu_id==id) return j;
    return -1;
}
static unsigned hsakmt_get_gfxv_by_node_id(unsigned node) {
    gfx_calls[gfx_count++]=node;return node==7 ? 950:1200;
}
static int fmm_set_memory_policy(unsigned id, int a, int b, uintptr_t base, uint64_t size, unsigned flag) {
    assert(base==4096 && size==4096);ids[calls]=id;flags[calls++]=flag;
    return gpu_mem_find_by_gpu_id(id)<0 || id==fail_id;
}
static int exercise(unsigned num_of_sysfs_nodes) {
    unsigned i, mfma_high_precision_mode=16;
    int gpu_mem_id=0,ret=0;
'''
SUFFIX = r'''
    return 0;
set_memory_policy_failed:
    return ret;
}
int main(void) {
    /* Kernel ordering differs from visible node ordering; the middle GPU is hidden. */
    gpu_count=2;gpu_mem[0].gpu_id=11;gpu_mem[0].node_id=7;
    gpu_mem[1].gpu_id=33;gpu_mem[1].node_id=2;
    process_apertures[0].gpu_id=22;process_apertures[0].gpuvm_limit=1;
    process_apertures[1].gpu_id=11;process_apertures[1].gpuvm_limit=1;
    process_apertures[2].gpu_id=33;process_apertures[2].gpuvm_limit=1;
    assert(exercise(3)==0);
    assert(calls==2 && ids[0]==11 && ids[1]==33);
    assert(gfx_calls[0]==7 && gfx_calls[1]==2 && flags[0]==16 && flags[1]==0);
    calls=gfx_count=0;fail_id=33;
    assert(exercise(3)==1); /* Preserve a real error on an accessible GPU. */
    assert(calls==2);
    calls=gfx_count=0;fail_id=0;process_apertures[1].gpuvm_limit=0;
    assert(exercise(3)==0 && calls==1 && ids[0]==33);
    puts("PASS hidden devices skipped, selected node mapping preserved, real failures propagated");
}
'''
old = subprocess.check_output(['git','-C',str(SOURCE),'show','4e4d7abeb786e903ac2a46815ae2565de331cbd4:libhsakmt/src/fmm.c'],text=True)
new = (SOURCE / 'libhsakmt/src/fmm.c').read_text()
with tempfile.TemporaryDirectory(prefix='filtered-policy-') as directory:
    directory=Path(directory)
    for name, text, expected in [('baseline',old,False),('patched',new,True)]:
        path=directory/(name+'.c');binary=directory/name
        path.write_text(PREFIX+policy_loop(text)+SUFFIX)
        subprocess.run(['/usr/bin/gcc','-std=gnu11','-O0',str(path),'-o',str(binary)],check=True)
        result=subprocess.run([str(binary)],capture_output=True,text=True)
        assert (result.returncode==0)==expected,(name,result.returncode,result.stderr)
        print(name, 'exit='+str(result.returncode),result.stdout.strip())
print('Regression reproduced on original loop and resolved by patched loop; no GPU invoked.')

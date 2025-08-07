
#ifndef OMP_H
#define OMP_H

// Minimal definitions to allow compilation to proceed
// These are not functional OpenMP implementations for WebAssembly,
// but serve to satisfy the compiler's need for omp.h.

#ifdef __cplusplus
extern "C" {
#endif

typedef void *omp_lock_t;
typedef void *omp_nest_lock_t;

int omp_get_thread_num(void);
int omp_get_num_threads(void);
int omp_get_max_threads(void);
int omp_get_num_procs(void);
int omp_in_parallel(void);
void omp_set_num_threads(int num_threads);
void omp_set_dynamic(int dynamic_threads);
int omp_get_dynamic(void);
int omp_get_nested(void);
void omp_init_lock(omp_lock_t *lock);
void omp_destroy_lock(omp_lock_t *lock);
void omp_set_lock(omp_lock_t *lock);
void omp_unset_lock(omp_lock_t *lock);
int omp_test_lock(omp_lock_t *lock);
void omp_init_nest_lock(omp_nest_lock_t *lock);
void omp_destroy_nest_lock(omp_nest_lock_t *lock);
void omp_set_nest_lock(omp_nest_lock_t *lock);
void omp_unset_nest_lock(omp_nest_lock_t *lock);
int omp_test_nest_lock(omp_nest_lock_t *lock);
double omp_get_wtime(void);
double omp_get_wtick(void);
void omp_set_schedule(int kind, int chunk_size);
void omp_get_schedule(int *kind, int *chunk_size);
int omp_get_thread_limit(void);
int omp_get_active_level(void);
int omp_get_level(void);
int omp_get_ancestor_thread_num(int level);
int omp_get_team_size(int level);
int omp_get_cancellation(void);
void omp_set_default_device(int device_num);
int omp_get_default_device(void);
int omp_get_num_devices(void);
int omp_get_num_teams(void);
int omp_get_team_num(void);
int omp_is_initial_device(void);
int omp_get_public_memory(void **ptr);
int omp_get_private_memory(void **ptr);
int omp_get_unified_memory(void **ptr);
int omp_get_initial_device(void);
int omp_get_max_task_priority(void);

#ifdef __cplusplus
}
#endif

#endif // OMP_H

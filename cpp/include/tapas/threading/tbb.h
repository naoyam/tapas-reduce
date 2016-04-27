#ifndef _TAPAS_THREADING_TBB_H_
#define _TAPAS_THREADING_TBB_H_

/** 
 * @file tapas/thread/tbb.h
 */
#include <type_traits>
#include <tbb/task_group.h>

namespace tapas {
namespace threading {

/**
 * Intel TBB
 */
class IntelTBB { // NOTE: we avoid naming the class "TBB", because it often conflicts configuring macros.
 public:
  //typedef myth_thread_t tid_t;
  
  static const constexpr bool Concurrent = false;
  static const constexpr bool Preemptive = false;

  static const char *name() { return "Intel TBB"; }

  using Task = tbb::task;

  static void init() { }

  template<class F>
  class CallableTask : public Task {
    F f_;
   public:
    CallableTask(F f) : f_(f) {}
    virtual void *execute() override {
      f_();
      return nullptr;
    }
  };

  class TaskGroup : public tbb::task_group {
   public:
    template<class F>
    void createTask(F f) {
      run(f);
    }

    //void wait() {
    //  wait();
    //}
  };
};

}
}

#endif  // _TAPAS_THREADING_TBB_H_




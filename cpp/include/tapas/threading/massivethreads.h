#ifndef _TAPAS_THREADING_MASSIVETHREADS_H_
#define _TAPAS_THREADING_MASSIVETHREADS_H_

#include <vector>

/** 
 * @file tapas/thread/massivethread.h
 */
#include <myth.h>

namespace tapas {
namespace threading {

/**
 * MassiveThreads
 */
class MassiveThreads {
 public:
  typedef myth_thread_t tid_t;
  
  static const constexpr bool Preenptive = false;
  
  class Task {
   private:
    tid_t tid_;
    
    static void* invoke(void * arg) {
      reinterpret_cast<Task*>(arg)->execute();
      return nullptr;
    }
    
   protected:
    virtual void *execute() = 0;
    
   public:
    void run() {
      tid_ = myth_create(&Task::invoke, reinterpret_cast<void*>(this));
    }

    inline tid_t id() const { return tid_; }
  };
  
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

  class TaskGroup {
    std::vector<Task*> tasks_;
   public:
    TaskGroup() : tasks_() { }

    template<class F>
    void create_task(F f) {
      auto *t = new CallableTask<F>(f);
      t->run();
      tasks_.push_back(t);
    }

    void wait() {
      for (auto &t : tasks_) {
        myth_join(t->id(), nullptr);
      }

      tasks_.clear();
    }
  };
  
  void yield() {
    myth_yield(0);
  }
  
};

}
}

#endif  // _TAPAS_THREADING_MASSIVETHREADS_H_




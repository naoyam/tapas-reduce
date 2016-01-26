#ifndef _TAPAS_THREADING_SERIAL_H_
#define _TAPAS_THREADING_SERIAL_H_

namespace tapas {
namespace threading {

class Serial {
 public:
  typedef int tid_t; // dummy

  static const constexpr bool Concurrent = false;
  static const constexpr bool Preemptive = false;

  static void init() { }

  class Task {
   private:
   protected:
    virtual void execute() = 0;
   public:
    void run() {
      this->execute();
    }
    
    inline tid_t id() const { return 0; }
  };

  template<class F>
  class CallableTask : public Task {
    F f_;
   public:
    CallableTask(F f) : f_(f) { }
    virtual void execute() override {
      f_();
    }
  };
  
  class TaskGroup {
   public:
    TaskGroup() { }
    template<class F>
    void createTask(F f) {
      auto *t = new CallableTask<F>(f);
      t->run();
    }

    void wait() { }
  };

  static void yield() { }
  
  template<class F>
  static void run(F f) {
    CallableTrask(f).execute();
  }
};

}
}

#endif // _TAPAS_THREADING_SERIAL_H_

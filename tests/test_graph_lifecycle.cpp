#include "graph_lifecycle.hpp"
#include <array>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
#include <fstream>
using namespace ts::lifecycle;
void check(bool b){if(!b)throw std::runtime_error("lifecycle assertion");}
void trace(std::initializer_list<Event> events,std::initializer_list<const char*> rules){State s;auto r=rules.begin();for(auto e:events){auto out=reduce(s,e);check(r!=rules.end()&&std::string(out.rule)==*r++);s=out.state;}check(r==rules.end());}
int main(){
 trace({Event::Begin,Event::Allocate,Event::Compute,Event::Completed,Event::Read},{"R1-begin-reset","R3-allocated","R6-compute","R9-computed","R12-read"});
 trace({Event::Read,Event::Begin,Event::Read,Event::Allocate,Event::Read},{"R2-idle-order","R1-begin-reset","R5-building-order","R3-allocated","R8-allocated-order"});
 trace({Event::Begin,Event::Allocate,Event::PreflightFail,Event::Read,Event::Begin,Event::Allocate,Event::Compute,Event::Completed,Event::Read},{"R1-begin-reset","R3-allocated","R7-preflight-failed","R14-failed-retains-first","R1-begin-reset","R3-allocated","R6-compute","R9-computed","R12-read"});
 trace({Event::Begin,Event::Allocate,Event::Compute,Event::Begin,Event::WorkerFailure,Event::Completed,Event::Read},{"R1-begin-reset","R3-allocated","R6-compute","R15-no-begin-during-compute","R10-worker-failed","R14-failed-retains-first","R14-failed-retains-first"});
 trace({Event::Begin,Event::AllocationFail,Event::WorkerFailure,Event::Begin},{"R1-begin-reset","R4-allocation-failed","R14-failed-retains-first","R1-begin-reset"});
 trace({Event::Begin,Event::Allocate,Event::Compute,Event::Completed,Event::UploadFail,Event::Read,Event::Begin},{"R1-begin-reset","R3-allocated","R6-compute","R9-computed","R16-upload-failed","R14-failed-retains-first","R1-begin-reset"});
 trace({Event::Begin,Event::Allocate,Event::Compute,Event::Completed,Event::Compute,Event::Read},{"R1-begin-reset","R3-allocated","R6-compute","R9-computed","R17-computed-mutation-invalidates","R14-failed-retains-first"});
 trace({Event::Begin,Event::Allocate,Event::Allocate,Event::Read,Event::Begin,Event::AllocationFail,Event::Read},{"R1-begin-reset","R3-allocated","R18-reallocate-invalidates","R14-failed-retains-first","R1-begin-reset","R4-allocation-failed","R14-failed-retains-first"});
 trace({Event::Begin,Event::Allocate,Event::Compute,Event::Completed,Event::BeginFail,Event::Read},{"R1-begin-reset","R3-allocated","R6-compute","R9-computed","R21-begin-failed","R14-failed-retains-first"});
 trace({Event::Begin,Event::Allocate,Event::Compute,Event::Completed,Event::AllocationFail,Event::Read},{"R1-begin-reset","R3-allocated","R6-compute","R9-computed","R22-allocation-failed-invalidates","R14-failed-retains-first"});
 trace({Event::Begin,Event::Allocate,Event::Compute,Event::Completed,Event::PreflightFail,Event::Read,Event::Begin,Event::Allocate,Event::Compute,Event::Completed,Event::Read},{"R1-begin-reset","R3-allocated","R6-compute","R9-computed","R24-repeat-preflight-invalidates","R14-failed-retains-first","R1-begin-reset","R3-allocated","R6-compute","R9-computed","R12-read"});
 trace({Event::Begin,Event::Allocate,Event::AllocationFail,Event::Read,Event::Begin},{"R1-begin-reset","R3-allocated","R25-reallocation-failed","R14-failed-retains-first","R1-begin-reset"});
 trace({Event::Begin,Event::ValidationRejected,Event::Allocate,Event::ValidationRejected,Event::Upload,Event::Compute,Event::Completed,Event::Read},{"R1-begin-reset","R27-premutation-rejection-preserves","R3-allocated","R27-premutation-rejection-preserves","R20-upload","R6-compute","R9-computed","R12-read"});
 trace({Event::Begin,Event::Allocate,Event::Compute,Event::Completed,Event::ValidationRejected,Event::Read},{"R1-begin-reset","R3-allocated","R6-compute","R9-computed","R26-readable-rejection-invalidates","R14-failed-retains-first"});
 std::vector<State> states={State{Idle{}},State{Building{}},State{Allocated{}},State{Computing{}},State{Computed{}},State{Failed{Error::Allocation}},State{Failed{Error::Preflight}},State{Failed{Error::Worker}},State{Failed{Error::Upload}},State{Failed{Error::Order}}};
 size_t pairs=0;std::ofstream graph("transitions.json");graph<<"[";
 for(size_t i=0;i<states.size();++i)for(int j=0;j<int(Event::Count);++j){auto e=Event(j);auto out=reduce(states[i],e);check(std::string(out.rule)!="R0-unreachable");check(out.effect!=Effect::ReturnOutput||(std::holds_alternative<Computed>(states[i].phase)&&e==Event::Read));
 if(std::holds_alternative<Computed>(states[i].phase)&&(e==Event::ValidationRejected||e==Event::BeginFail||e==Event::AllocationFail||e==Event::UploadFail||e==Event::PreflightFail||e==Event::Allocate||e==Event::Upload||e==Event::Compute)){check(!std::holds_alternative<Computed>(out.state.phase));check(reduce(out.state,Event::Read).effect==Effect::Reject);}
 if(auto f=std::get_if<Failed>(&states[i].phase);f&&e!=Event::Begin&&e!=Event::BeginFail){auto next=std::get_if<Failed>(&out.state.phase);check(next&&next->error==f->error);}
 if(e==Event::Begin&&!std::holds_alternative<Computing>(states[i].phase))check(std::holds_alternative<Building>(out.state.phase));
 if(pairs++){graph<<",";}
 graph<<"{\"state\":"<<i<<",\"event\":"<<j<<",\"next\":"<<out.state.phase.index()<<",\"rule\":\""<<out.rule<<"\",\"effect\":"<<int(out.effect)<<"}";
 }graph<<"]\n";
 // Exhaustive finite reachability: every phase plus every modeled first error.
 std::array<bool,10> seen{};seen[0]=true;bool changed=true;while(changed){changed=false;for(size_t i=0;i<states.size();++i)if(seen[i])for(int j=0;j<int(Event::Count);++j){auto n=reduce(states[i],Event(j)).state;size_t k=n.phase.index();if(auto f=std::get_if<Failed>(&n.phase))k=f->error==Error::Allocation?5:f->error==Error::Preflight?6:f->error==Error::Worker?7:f->error==Error::Upload?8:9;if(!seen[k]){seen[k]=true;changed=true;}}}for(bool b:seen)check(b);
 // Each phase terminates or waits for a caller/worker event; Computing has no retry/timeout.
 for(auto s:states){if(std::holds_alternative<Computing>(s.phase))check(std::holds_alternative<Computed>(reduce(s,Event::Completed).state.phase));else check(std::holds_alternative<Building>(reduce(s,Event::Begin).state.phase));}
 State a{Failed{Error::Worker}},b{Computed{}};a=reduce(a,Event::Begin).state;check(std::holds_alternative<Computed>(b.phase));check(reduce(b,Event::Read).effect==Effect::ReturnOutput);
 std::cout<<"PASS traces=14 exhaustive-state-event-pairs="<<pairs<<" reachable=10 invariants/read/reset/first-error/isolation/liveness\n";
}

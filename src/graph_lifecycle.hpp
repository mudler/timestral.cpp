#pragma once
#include <cstdint>
#include <variant>
#include <type_traits>
namespace ts::lifecycle {
struct Idle{}; struct Building{}; struct Allocated{}; struct Computing{}; struct Computed{};
enum class Error { None, Order, Preflight, Worker, Allocation, Upload };
struct Failed { Error error; };
using Phase=std::variant<Idle,Building,Allocated,Computing,Computed,Failed>;
struct State { Phase phase=Idle{}; }; // one instance belongs to one GraphRunner
// Begin constructs a fresh run. No asynchronous events survive compute return.
enum class Event { Begin, Allocate, AllocationFail, Compute, PreflightFail, Completed, WorkerFailure, Read, Upload, UploadFail, BeginFail, ValidationRejected, Count };
enum class Effect { None, ResetRun, Allocate, Launch, ReturnOutput, Reject };
struct Result { State state; Effect effect; const char* rule; };
template<class> inline constexpr bool never=false;
inline Result reduce(State s,Event e){
 if(e==Event::BeginFail && !std::holds_alternative<Computing>(s.phase))return {State{Failed{Error::Allocation}},Effect::Reject,"R21-begin-failed"};
 if(e==Event::PreflightFail && std::holds_alternative<Computed>(s.phase))return {State{Failed{Error::Preflight}},Effect::Reject,"R24-repeat-preflight-invalidates"};
 if(e==Event::AllocationFail && std::holds_alternative<Allocated>(s.phase))return {State{Failed{Error::Allocation}},Effect::Reject,"R25-reallocation-failed"};
 if(e==Event::AllocationFail && std::holds_alternative<Computed>(s.phase))return {State{Failed{Error::Allocation}},Effect::Reject,"R22-allocation-failed-invalidates"};
 if(e==Event::Begin && std::holds_alternative<Computing>(s.phase))return {s,Effect::Reject,"R15-no-begin-during-compute"};
 if(e==Event::Begin)return {State{Building{}},Effect::ResetRun,"R1-begin-reset"};
 if(e==Event::ValidationRejected){if(std::holds_alternative<Computed>(s.phase))return {State{Failed{Error::Order}},Effect::Reject,"R26-readable-rejection-invalidates"};return {s,Effect::Reject,"R27-premutation-rejection-preserves"};}
 // Modifying attempts cannot retain a previously readable result.
 if(e==Event::UploadFail && !std::holds_alternative<Computing>(s.phase) && !std::holds_alternative<Failed>(s.phase))return {State{Failed{Error::Upload}},Effect::Reject,"R16-upload-failed"};
 if((e==Event::Upload||e==Event::Compute||e==Event::Allocate) && std::holds_alternative<Computed>(s.phase))return {State{Failed{Error::Order}},Effect::Reject,"R17-computed-mutation-invalidates"};
 if(e==Event::Allocate && std::holds_alternative<Allocated>(s.phase))return {State{Failed{Error::Order}},Effect::Reject,"R18-reallocate-invalidates"};
 return std::visit([&](auto p)->Result {
  using T=decltype(p);
  if constexpr(std::is_same_v<T,Idle>){
   switch(e){
    case Event::Begin: break;
    case Event::Allocate: case Event::AllocationFail: case Event::Compute: case Event::PreflightFail: case Event::Completed: case Event::WorkerFailure: case Event::Read: return {s,Effect::Reject,"R2-idle-order"};
    case Event::BeginFail:return {s,Effect::Reject,"R23-begin-fail-order"};
    case Event::Upload: case Event::UploadFail:return {s,Effect::Reject,"R19-upload-order"};
    case Event::ValidationRejected:return {s,Effect::Reject,"R0-unreachable-validation"};
    case Event::Count:return {s,Effect::Reject,"R0-invalid-event"};
   }
  }else if constexpr(std::is_same_v<T,Building>){
   switch(e){
    case Event::Begin:break;
    case Event::Allocate:return {State{Allocated{}},Effect::Allocate,"R3-allocated"};
    case Event::AllocationFail:return {State{Failed{Error::Allocation}},Effect::Reject,"R4-allocation-failed"};
    case Event::Compute: case Event::PreflightFail: case Event::Completed: case Event::WorkerFailure: case Event::Read:return {s,Effect::Reject,"R5-building-order"};
    case Event::BeginFail:return {s,Effect::Reject,"R23-begin-fail-order"};
    case Event::Upload: case Event::UploadFail:return {s,Effect::Reject,"R19-upload-order"};
    case Event::ValidationRejected:return {s,Effect::Reject,"R0-unreachable-validation"};
    case Event::Count:return {s,Effect::Reject,"R0-invalid-event"};
   }
  }else if constexpr(std::is_same_v<T,Allocated>){
   if(e==Event::Upload)return {s,Effect::None,"R20-upload"};
   switch(e){
    case Event::Begin:break;
    case Event::Compute:return {State{Computing{}},Effect::Launch,"R6-compute"};
    case Event::PreflightFail:return {State{Failed{Error::Preflight}},Effect::Reject,"R7-preflight-failed"};
    case Event::Allocate: case Event::AllocationFail: case Event::Completed: case Event::WorkerFailure: case Event::Read:return {s,Effect::Reject,"R8-allocated-order"};
    case Event::BeginFail:return {s,Effect::Reject,"R23-begin-fail-order"};
    case Event::Upload: case Event::UploadFail:return {s,Effect::Reject,"R19-upload-order"};
    case Event::ValidationRejected:return {s,Effect::Reject,"R0-unreachable-validation"};
    case Event::Count:return {s,Effect::Reject,"R0-invalid-event"};
   }
  }else if constexpr(std::is_same_v<T,Computing>){
   switch(e){
    case Event::Begin:break;
    case Event::Completed:return {State{Computed{}},Effect::None,"R9-computed"};
    case Event::WorkerFailure:return {State{Failed{Error::Worker}},Effect::Reject,"R10-worker-failed"};
    case Event::Allocate: case Event::AllocationFail: case Event::Compute: case Event::PreflightFail: case Event::Read:return {s,Effect::Reject,"R11-computing-order"};
    case Event::BeginFail:return {s,Effect::Reject,"R23-begin-fail-order"};
    case Event::Upload: case Event::UploadFail:return {s,Effect::Reject,"R19-upload-order"};
    case Event::ValidationRejected:return {s,Effect::Reject,"R0-unreachable-validation"};
    case Event::Count:return {s,Effect::Reject,"R0-invalid-event"};
   }
  }else if constexpr(std::is_same_v<T,Computed>){
   switch(e){
    case Event::Begin:break;
    case Event::Read:return {s,Effect::ReturnOutput,"R12-read"};
    case Event::Allocate: case Event::AllocationFail: case Event::Compute: case Event::PreflightFail: case Event::Completed: case Event::WorkerFailure:return {s,Effect::Reject,"R13-computed-order"};
    case Event::BeginFail:return {s,Effect::Reject,"R23-begin-fail-order"};
    case Event::Upload: case Event::UploadFail:return {s,Effect::Reject,"R19-upload-order"};
    case Event::ValidationRejected:return {s,Effect::Reject,"R0-unreachable-validation"};
    case Event::Count:return {s,Effect::Reject,"R0-invalid-event"};
   }
  }else if constexpr(std::is_same_v<T,Failed>){
   switch(e){
    case Event::Begin:break;
    case Event::Allocate: case Event::AllocationFail: case Event::Compute: case Event::PreflightFail: case Event::Completed: case Event::WorkerFailure: case Event::Read:return {s,Effect::Reject,"R14-failed-retains-first"};
    case Event::BeginFail:return {s,Effect::Reject,"R23-begin-fail-order"};
    case Event::Upload: case Event::UploadFail:return {s,Effect::Reject,"R19-upload-order"};
    case Event::ValidationRejected:return {s,Effect::Reject,"R0-unreachable-validation"};
    case Event::Count:return {s,Effect::Reject,"R0-invalid-event"};
   }
  }else static_assert(never<T>);
  return {s,Effect::Reject,"R0-unreachable"};
 },s.phase);
}
}

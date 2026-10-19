#include "fixture.hpp"
#include "predict_metric.hpp"
template<class F>void bad(F f){bool rejected=false;try{f();}catch(const std::exception&){rejected=true;}require(rejected,"invalid metric input accepted");}
int main(){try{
 using tstest::compare_predict_normalized_v1;
 std::vector<float> y={100001,100002,-99999,-99998},loc={100000,-100000},scale={1,2};
 std::vector<double> f64={100001.001,100002.002,-99998.999,-99997.998},math={100001.003,100002.001,-99998.998,-99997.999};
 auto a=compare_predict_normalized_v1(y,y,f64,math,loc,scale);require(a.ok,"exact metric");
 auto corrupt=y;corrupt[2]+=1;require(!compare_predict_normalized_v1(corrupt,y,f64,math,loc,scale).ok,"large offset hides normalized error");
 auto swapped=compare_predict_normalized_v1(y,y,math,f64,loc,scale);require(swapped.tol!=a.tol,"F64 defines frame and RMS");
 bad([&]{compare_predict_normalized_v1(y,y,{},math,loc,scale);});bad([&]{compare_predict_normalized_v1(y,y,f64,{},loc,scale);});
 bad([&]{compare_predict_normalized_v1(y,y,f64,math,loc,{0,2});});
 bad([&]{compare_predict_normalized_v1(y,y,f64,math,loc,{-1,2});});
 bad([&]{compare_predict_normalized_v1(y,y,f64,math,loc,{NAN,2});});
 bad([&]{compare_predict_normalized_v1({0},{0},{0},{0},{0},{1});});
 auto other=math;other[0]+=10;require(compare_predict_normalized_v1(y,y,f64,other,loc,scale).tol>a.tol,"second companion contributes floor");
 auto first=f64;first[0]+=10;require(compare_predict_normalized_v1(y,y,first,math,loc,scale).tol>a.tol,"first companion contributes floor");
 return 0;
 }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}

#include "masks.hpp"
#include <algorithm>
#include <limits>
namespace ts {
namespace {
size_t product(size_t a,size_t b){if(b&&a>std::numeric_limits<size_t>::max()/b)throw Error("mask dimensions overflow");return a*b;}
}
Masks build_masks(const Config& c,const std::vector<int8_t>& observed,size_t R,size_t T,size_t h,const std::vector<int64_t>& groups,const std::vector<int64_t>& roles){
 if(!R||!T||!h||!c.patch_size||!c.block_size||T%c.patch_size||observed.size()!=product(R,T)||groups.size()!=R||(!roles.empty()&&roles.size()!=R))throw Error("invalid mask dimensions");
 for(auto v:observed)if(v!=0&&v!=1)throw Error("observation mask must be boolean");
 for(auto v:roles)if(v<0||v>2)throw Error("invalid row role");
 bool cov=c.covariate_enabled&&!roles.empty()&&std::any_of(roles.begin(),roles.end(),[](int64_t v){return v!=0;});
 size_t N=T/c.patch_size;
 if((cov&&N<=h)||N>size_t(INT64_MAX)/2||h>size_t(INT64_MAX)-N)throw Error("invalid mask horizon");
 size_t L=cov?N-h:N,S=cov?2*N:N+h,ndec=cov?N:h;
 Masks a;a.obs.assign(R*N,0);a.key_valid.resize(R*N);
 for(size_t r=0;r<R;++r){bool started=false;for(size_t p=0;p<N;++p){bool o=false;for(size_t j=0;j<c.patch_size;++j)o|=observed[r*T+p*c.patch_size+j]!=0;a.obs[r*N+p]=o;started|=o;a.key_valid[r*N+p]=o||(c.attend_missing_patches&&started);}}
 auto unique=groups;std::sort(unique.begin(),unique.end());unique.erase(std::unique(unique.begin(),unique.end()),unique.end());
 a.inverse.resize(R);a.first_group.assign(unique.size(),N);
 for(size_t r=0;r<R;++r){size_t g=std::lower_bound(unique.begin(),unique.end(),groups[r])-unique.begin();a.inverse[r]=g;for(size_t p=0;p<N;++p)if(a.obs[r*N+p]){a.first_group[g]=std::min<int64_t>(a.first_group[g],p);break;}}
 for(auto& v:a.first_group)if(v==int64_t(N))v=0;
 a.positions.resize(product(R,S));
 for(size_t r=0;r<R;++r)for(size_t p=0;p<S;++p)a.positions[r*S+p]=int64_t(cov?p%N:p)-a.first_group[a.inverse[r]];
 a.time.enc_len=N;a.time.allowed.assign(product(product(R,S),S),0);
 if(cov){a.anchor.resize(R*N);a.active.resize(R*N);for(size_t r=0;r<R;++r)for(size_t p=0;p<N;++p){a.active[r*N+p]=p>=L;a.anchor[r*N+p]=p>=L?L+c.block_size*((p-L)/c.block_size):0;}}
 for(size_t r=0;r<R;++r)for(size_t q=0;q<S;++q)for(size_t k=0;k<S;++k){bool yes=false;
  if(q<N&&k<N)yes=(a.key_valid[r*N+k]&&((cov&&roles[r]==2)||k<=q))||q==k;
  else if(q>=N&&k<N)yes=a.key_valid[r*N+k]&&(!cov||(a.active[r*N+q-N]&&int64_t(k)<a.anchor[r*N+q-N]));
  else if(q>=N&&k>=N)yes=q==k||(!cov?k<=q:(k<=q&&a.active[r*N+q-N]&&a.active[r*N+k-N]&&a.anchor[r*N+q-N]==a.anchor[r*N+k-N]));
  a.time.allowed[(r*S+q)*S+k]=yes;
 }
 if(R>1){a.variate.resize(product(product(S,R),R));for(size_t p=0;p<N+ndec;++p)for(size_t q=0;q<R;++q)for(size_t k=0;k<R;++k){bool forbid=cov&&((roles[q]!=0&&roles[k]==0)||(roles[q]==2&&roles[k]!=2));a.variate[(p*R+q)*R+k]=q==k||(groups[q]==groups[k]&&(p>=N||a.key_valid[k*N+p])&&!forbid);}}
 return a;
}
}

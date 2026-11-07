// Neighbour-context coder for the shared-SH index arrays (app_mapping / app_residual_mapping).
// Reuses the tested adaptive binary arithmetic coder of stage2/codec/native.cpp (same trick as edge.cpp):
// build with -I <stage2>/codec.  Decodable: the decoder only looks at codes of EARLIER vertices (already decoded)
// that are mesh neighbours, so topology + vertex order must be decoded first (true in the delivered asset).
//
// Per vertex i, in index order: cand = distinct codes of earlier neighbours (first-appearance order, <= 32).
//   m = |cand| > 0  ->  hit bit (context: m)        hit: slot index via 5-bit prefix tree (context: m)
//   miss or m == 0  ->  raw code via nbits prefix tree (adaptive order-0 model of the K codes)
#define main xyz_program_main
#include "native.cpp"
#undef main
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
namespace py=pybind11;
static const int MAXC=32;
struct Model{
 int nbits;std::vector<Count> hit,slot,raw;
 Model(U K):nbits(0){while((U(1)<<nbits)<K)nbits++;hit.resize(MAXC+1);slot.resize((MAXC+1)*64);raw.resize(size_t(1)<<(nbits+1));}
};
template<class F> static void prefix(int nbits,std::vector<Count>&c,size_t base,U&x,F&&bit){int p=1;U v=0;for(int b=nbits-1;b>=0;b--){int k=bit(c[base+p],(x>>b)&1);v=(v<<1)|k;p=p*2+k;}x=v;}
static int candidates(size_t i,const int64_t*ptr,const int64_t*dst,const I*code,U*cand){int m=0;
 for(int64_t j=ptr[i];j<ptr[i+1];j++){if(size_t(dst[j])>=i)throw std::runtime_error("neighbour not earlier than vertex");U c=U(code[dst[j]]);bool seen=false;
  for(int k=0;k<m;k++)if(cand[k]==c){seen=true;break;}if(!seen){if(m==MAXC)throw std::runtime_error("more than 32 distinct earlier neighbour codes");cand[m++]=c;}}
 return m;}
py::bytes encode(py::array_t<I,py::array::c_style|py::array::forcecast> code,py::array_t<int64_t,py::array::c_style|py::array::forcecast> ptr,py::array_t<int64_t,py::array::c_style|py::array::forcecast> dst,U K){
 size_t n=code.size();if(size_t(ptr.size())!=n+1)throw std::runtime_error("ptr size");
 const I*c=code.data();const int64_t*p=ptr.data();const int64_t*d=dst.data();Model M(K);Encoder enc;U cand[MAXC];
 auto bit=[&](Count&ct,int b){enc.code(b,ct);return b;};
 for(size_t i=0;i<n;i++){if(c[i]<0||U(c[i])>=K)throw std::runtime_error("code out of range");
  int m=candidates(i,p,d,c,cand);int s=-1;for(int k=0;k<m;k++)if(cand[k]==U(c[i])){s=k;break;}
  if(m>0){enc.code(s>=0,M.hit[m]);if(s>=0){U x=U(s);prefix(5,M.slot,size_t(m)*64,x,bit);continue;}}
  U x=U(c[i]);prefix(M.nbits,M.raw,0,x,bit);}
 enc.finish();return py::bytes(reinterpret_cast<const char*>(enc.out.v.data()),enc.out.v.size());}
py::array_t<I> decode(py::bytes data,py::array_t<int64_t,py::array::c_style|py::array::forcecast> ptr,py::array_t<int64_t,py::array::c_style|py::array::forcecast> dst,U K,size_t n){
 std::string s=data;std::vector<uint8_t> bytes(s.begin(),s.end());if(size_t(ptr.size())!=n+1)throw std::runtime_error("ptr size");
 const int64_t*p=ptr.data();const int64_t*d=dst.data();Model M(K);Decoder dec(bytes);U cand[MAXC];
 py::array_t<I> out(n);I*c=out.mutable_data();
 auto bit=[&](Count&ct,int){return dec.code(ct);};
 for(size_t i=0;i<n;i++){int m=candidates(i,p,d,c,cand);
  if(m>0&&dec.code(M.hit[m])){U x=0;prefix(5,M.slot,size_t(m)*64,x,bit);if(int(x)>=m)throw std::runtime_error("slot out of range");c[i]=I(cand[x]);continue;}
  U x=0;prefix(M.nbits,M.raw,0,x,bit);c[i]=I(x);}
 return out;}
PYBIND11_MODULE(_index_ctx,m){m.def("encode",&encode,"code int32[N], ptr int64[N+1], dst int64[E] (earlier neighbours only), K -> bytes");m.def("decode",&decode,"bytes, ptr, dst, K, N -> int32[N]");}

// DERIVED from stage2/legacy_hierarchy/native_qem.cpp (Wenhao's error-adaptive hierarchy builder).
// Change: the geometric error term is measured in PIXELS instead of median-edge units.
//   score = max( cluster_position_radius * cluster_pixel_scale , cluster_appearance_radius * appearance_unit )
// where pixel_scale[v] = max over training views that render v of (focal_px / depth)   [px per world unit]
// (passed in, converted to normalised units here) and clusters carry the max pixel_scale of their members.
// appearance_scale[v] (optional, px per appearance unit, cluster max) replaces the global appearance_unit per vertex.
// With pixel_scale == None and appearance_unit == 1 the behaviour is bit-identical to the original
// (pixel_scale defaults to 1/median_edge).  v5: target_vertices (collapse in cost order until that many
// vertices remain; takes precedence over target_faces).  build_error also accepts target_faces (collapse in cost
// order until that many faces remain) so face counts can be matched across criteria.
// Endpoint QEM: compact native adjacency, immutable positions and stable face IDs.
// Boundary/nonmanifold locks, full vertex-link test, duplicate/normal safeguards.
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <queue>
#include <stdexcept>
#include <tuple>
#include <vector>
namespace py=pybind11;
using Face=std::array<int,3>;
using Vec=std::array<double,3>;
static Vec sub(Vec a,Vec b){return {a[0]-b[0],a[1]-b[1],a[2]-b[2]};}
static Vec cross(Vec a,Vec b){return {a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]};}
static double dot(Vec a,Vec b){return a[0]*b[0]+a[1]*b[1]+a[2]*b[2];}
static void unique(std::vector<int>& a){std::sort(a.begin(),a.end());a.erase(std::unique(a.begin(),a.end()),a.end());}
static bool contains(Face f,int v){return f[0]==v||f[1]==v||f[2]==v;}
static uint64_t edge(int a,int b){if(a>b)std::swap(a,b);return (uint64_t(uint32_t(a))<<32)|uint32_t(b);}
#include "collapse_policy.h"
struct Item {double cost;int u,v;uint64_t vu,vv;};
struct Greater {bool operator()(const Item&a,const Item&b) const {
 return std::tie(a.cost,a.u,a.v,a.vu,a.vv)>std::tie(b.cost,b.u,b.v,b.vu,b.vv);}};
struct NativeQEM {
 std::vector<Vec> xyz;
 std::vector<Face> faces;
 std::vector<std::vector<int>> incident;
 std::vector<std::array<double,10>> q;
 std::vector<uint8_t> alive_v,alive_f,locked;
 std::vector<uint64_t> version;
 std::priority_queue<Item,std::vector<Item>,Greater> heap;
 std::vector<std::array<int,2>> collapses;
 std::string policy;bool area_weighted=false;
 std::vector<int> roles;
 // Error-adaptive mode: conservative cluster-radius bounds, NOT face budgets.
 std::vector<double> radius,appearance_radius;
 std::vector<std::vector<double>> appearance;
 double edge_unit=1.,extent=1.;
 // Pixel-space terms (see header comment). pixel_scale is per surviving cluster, in px per normalised unit.
 std::vector<double> pixel_scale,app_scale;double appearance_unit=1.,pending_pixel_scale=0.,pending_app_scale=0.,max_accepted_cost=0.;
 std::vector<int> pending_drop,pending_orphans;
 double pending_radius=0.,pending_appearance_radius=0.;
 int64_t duplicate_faces_removed=0,orphan_vertices_merged=0;
 std::array<int64_t,4> initial_roles={0,0,0,0};
 std::map<std::string,int64_t> rejected;
 int64_t accepted_interior=0,accepted_chain=0,accepted_into_legacy_locked=0;
 int64_t accepted_boundary_chain=0,accepted_nonmanifold_chain=0;
 int count=0; size_t peak_heap=0,rebuilds=0; double cos_limit=std::cos(80.*3.141592653589793/180.);
 Vec normal(Face f)const{return cross(sub(xyz[f[1]],xyz[f[0]]),sub(xyz[f[2]],xyz[f[0]]));}
 NativeQEM(py::array_t<double,py::array::c_style|py::array::forcecast> vertices,
           py::array_t<int64_t,py::array::c_style|py::array::forcecast> fs, py::object progress,
           std::string mode="conservative",double max_angle=80.,bool weighted=false):policy(mode),area_weighted(weighted){
  if(policy!="conservative"&&policy!="anchored"&&policy!="stratified"&&policy!="error-adaptive")throw std::runtime_error("Unknown collapse policy");
  if(!(max_angle>0&&max_angle<90))throw std::runtime_error("Normal angle must be between 0 and 90");
  cos_limit=std::cos(max_angle*3.141592653589793/180.);
  if(vertices.ndim()!=2||vertices.shape(1)!=3||fs.ndim()!=2||fs.shape(1)!=3||fs.shape(0)==0)
   throw std::runtime_error("Expected vertices [V,3] and nonempty faces [F,3]");
  if(vertices.shape(0)>INT32_MAX||fs.shape(0)>INT32_MAX)throw std::runtime_error("Native backend uses signed 32-bit stable IDs");
  int nv=vertices.shape(0),nf=fs.shape(0);auto v=vertices.unchecked<2>();auto f=fs.unchecked<2>();
  xyz.resize(nv);incident.resize(nv);q.resize(nv);alive_v.assign(nv,1);alive_f.assign(nf,1);locked.assign(nv,0);version.assign(nv,0);faces.resize(nf);count=nf;
  Vec lo={INFINITY,INFINITY,INFINITY},hi={-INFINITY,-INFINITY,-INFINITY};
  for(int i=0;i<nv;++i)for(int j=0;j<3;++j){double x=v(i,j);if(!std::isfinite(x))throw std::runtime_error("Nonfinite vertex");lo[j]=std::min(lo[j],x);hi[j]=std::max(hi[j],x);}
  double scale=std::sqrt(dot(sub(hi,lo),sub(hi,lo)));if(!(scale>0))throw std::runtime_error("Zero extent");extent=scale;
  for(int i=0;i<nv;++i)for(int j=0;j<3;++j)xyz[i][j]=(v(i,j)-lo[j])/scale;
  std::vector<Face> keys;keys.reserve(nf);std::vector<uint64_t> edges;edges.reserve(size_t(nf)*3);
  for(int i=0;i<nf;++i){
   if(i%1000000==0){if(PyErr_CheckSignals()!=0)throw py::error_already_set();if(!progress.is_none()){
    py::dict row;row["phase"]="validate faces and accumulate quadrics";row["faces_processed"]=i;progress(row);}}
   for(int j=0;j<3;++j){auto id=f(i,j);if(id<0||id>=nv)throw std::runtime_error("Face index out of range");faces[i][j]=int(id);}
   Face key=faces[i];std::sort(key.begin(),key.end());
   if(key[0]==key[1]||key[1]==key[2])throw std::runtime_error("Repeated-vertex input face; no automatic repair");
   keys.push_back(key);Vec n=normal(faces[i]);double len=std::sqrt(dot(n,n));
   if(len<=1e-14)throw std::runtime_error("Degenerate input face; no automatic repair");
   for(double &a:n)a/=len;double p[4]={n[0],n[1],n[2],-dot(n,xyz[faces[i][0]])};
   for(int id:faces[i]){incident[id].push_back(i);int k=0;for(int a=0;a<4;++a)for(int b=a;b<4;++b)q[id][k++]+=(area_weighted?len*.5:1.)*p[a]*p[b];}
   for(int j=0;j<3;++j)edges.push_back(edge(faces[i][j],faces[i][(j+1)%3]));
  }
  std::sort(keys.begin(),keys.end());if(std::adjacent_find(keys.begin(),keys.end())!=keys.end())throw std::runtime_error("Duplicate input faces; no automatic repair");
  std::vector<Face>().swap(keys);
  if(!progress.is_none()){py::dict row;row["phase"]="sort edge incidence and lock nonmanifold vertices";progress(row);}
  std::sort(edges.begin(),edges.end());
  for(size_t a=0;a<edges.size();){size_t b=a+1;while(b<edges.size()&&edges[b]==edges[a])++b;
   if(b-a!=2){locked[edges[a]>>32]=1;locked[uint32_t(edges[a])]=1;}a=b;}
  // Connected incident face fan is required even when each edge has two faces.
  for(int u=0;u<nv;++u){if(locked[u]||incident[u].empty())continue;
   if(u%100000==0&&PyErr_CheckSignals()!=0)throw py::error_already_set();
   std::vector<int> seen={incident[u][0]};
   for(size_t i=0;i<seen.size();++i)for(int w:faces[seen[i]])if(w!=u)
    for(int fid:incident[w])if(contains(faces[fid],u)&&std::find(seen.begin(),seen.end(),fid)==seen.end())seen.push_back(fid);
   if(seen.size()!=incident[u].size())locked[u]=1;
  }
  edges.erase(std::unique(edges.begin(),edges.end()),edges.end());
  if(policy=="error-adaptive"){
   std::vector<double> lengths;lengths.reserve(edges.size());
   for(auto e:edges){auto d=sub(xyz[e>>32],xyz[uint32_t(e)]);lengths.push_back(std::sqrt(dot(d,d)));}
   auto mid=lengths.begin()+lengths.size()/2;std::nth_element(lengths.begin(),mid,lengths.end());edge_unit=*mid;
   if(!(edge_unit>0))throw std::runtime_error("Invalid median edge length");
   radius.assign(nv,0.);appearance_radius.assign(nv,0.);pixel_scale.assign(nv,1./edge_unit);app_scale.assign(nv,1.);
  }
  if(policy!="conservative"&&policy!="error-adaptive"){
   if(!progress.is_none()){py::dict row;row["phase"]="classify dynamic collapse roles";progress(row);}
   roles.resize(nv);CollapsePolicy checker{faces,incident};
   for(int i=0;i<nv;++i){roles[i]=checker.classify(i);++initial_roles[roles[i]];if(i%100000==0&&PyErr_CheckSignals()!=0)throw py::error_already_set();}
  }
  if(!progress.is_none()){py::dict row;row["phase"]="initialize priority queue";row["locked_vertices"]=std::count(locked.begin(),locked.end(),uint8_t(1));row["edges"]=edges.size();progress(row);}
  if(policy!="error-adaptive")for(auto e:edges){enqueue(e>>32,uint32_t(e));enqueue(uint32_t(e),e>>32);}
 }
 std::vector<int> neighbors(int u)const{
  std::vector<int> out;for(int fid:incident[u])for(int w:faces[fid])if(w!=u)out.push_back(w);unique(out);return out;}
 double cost(int u,int v)const{
  if(policy=="error-adaptive")return merge_error(u,v);
  double x[4]={xyz[v][0],xyz[v][1],xyz[v][2],1.},s=0;int k=0;
  for(int a=0;a<4;++a)for(int b=a;b<4;++b){s+=(a==b?1.:2.)*x[a]*x[b]*(q[u][k]+q[v][k]);++k;}return std::max(0.,s);}
 bool eligible(int u,int v)const{
  if(!alive_v[u]||!alive_v[v])return false;
  if(policy=="error-adaptive")return !incident[u].empty()&&!incident[v].empty();
  if(policy=="conservative")return !locked[u]&&!locked[v];
  return CollapsePolicy{faces,incident}.candidate(u,v,roles[u],policy);
 }
 void enqueue(int u,int v){if(eligible(u,v)){
  heap.push({cost(u,v),u,v,version[u],version[v]});peak_heap=std::max(peak_heap,heap.size());}}
 bool reject(const std::string& why){++rejected[why];return false;}
 double distance(int u,int v)const{auto d=sub(xyz[u],xyz[v]);return std::sqrt(dot(d,d));}
 double appearance_distance(int u,int v)const{
  double sum=0.;for(size_t j=0;j<appearance[u].size();++j){double d=appearance[u][j]-appearance[v][j];sum+=d*d;}return std::sqrt(sum);
 }
 double merge_error(int u,int v)const{
  return std::max(std::max(radius[v],radius[u]+distance(u,v))*std::max(pixel_scale[u],pixel_scale[v]),
                  std::max(appearance_radius[v],appearance_radius[u]+appearance_distance(u,v))*std::max(app_scale[u],app_scale[v]));
 }
 bool adaptive_legal(int u,int v){
  pending_drop.clear();pending_orphans.clear();
  auto changed=incident[u];std::sort(changed.begin(),changed.end());
  std::map<Face,int> keys;
  for(int fid:incident[v])if(!contains(faces[fid],u)){auto key=faces[fid];std::sort(key.begin(),key.end());keys[key]=fid;}
  bool shared=false;std::map<int,int> remaining;
  auto touched=neighbors(u);touched.push_back(u);touched.push_back(v);unique(touched);
  for(int w:touched)remaining[w]=incident[w].size();
  for(int fid:changed){
   Face old=faces[fid],next=old;for(int& w:next)if(w==u)w=v;
   bool degenerate=contains(old,v);shared=shared||degenerate;
   auto key=next;std::sort(key.begin(),key.end());
   bool drop=degenerate||keys.count(key);
   if(drop){pending_drop.push_back(fid);for(int w:old)--remaining[w];continue;}
   Vec a=normal(old),b=normal(next);double nb=std::sqrt(dot(b,b));
   if(nb<=1e-14||dot(a,b)<=cos_limit*std::sqrt(dot(a,a))*nb)return reject("geometry");
   keys[key]=fid;--remaining[u];++remaining[v];
  }
  if(!shared)return reject("edge_incidence");
  // Never erase a complete component: its receiver must still have a face.
  if(remaining[v]<=0)return reject("component_erasure");
  pending_radius=std::max(radius[v],radius[u]+distance(u,v));
  pending_appearance_radius=std::max(appearance_radius[v],appearance_radius[u]+appearance_distance(u,v));
  pending_pixel_scale=std::max(pixel_scale[u],pixel_scale[v]);pending_app_scale=std::max(app_scale[u],app_scale[v]);
  for(int w:touched)if(w!=u&&w!=v&&remaining[w]==0){
   pending_orphans.push_back(w);pending_pixel_scale=std::max(pending_pixel_scale,pixel_scale[w]);pending_app_scale=std::max(pending_app_scale,app_scale[w]);
   pending_radius=std::max(pending_radius,radius[w]+distance(w,v));
   pending_appearance_radius=std::max(pending_appearance_radius,appearance_radius[w]+appearance_distance(w,v));
  }
  return true;
 }
 bool legal(int u,int v){
  if(!eligible(u,v))return reject("source_role");
  if(policy=="error-adaptive")return adaptive_legal(u,v);
  std::vector<int> shared,opp;
  for(int fid:incident[u])if(contains(faces[fid],v)){shared.push_back(fid);for(int w:faces[fid])if(w!=u&&w!=v)opp.push_back(w);}
  if(shared.empty()||(policy=="conservative"&&shared.size()!=2))return reject("edge_incidence");unique(opp);
  auto nu=neighbors(u),nv=neighbors(v);std::vector<int> common;
  std::set_intersection(nu.begin(),nu.end(),nv.begin(),nv.end(),std::back_inserter(common));if(common!=opp)return reject("link_vertices");
  std::vector<Face> proposed;
  for(int fid:incident[u]){if(contains(faces[fid],v))continue;Face old=faces[fid],next=old;
   for(int &w:next)if(w==u)w=v;Face key=next;std::sort(key.begin(),key.end());
   if(std::find(proposed.begin(),proposed.end(),key)!=proposed.end())return reject("duplicate");proposed.push_back(key);
   // Equality here is exactly a common link edge (a,b) for triangles (u,a,b),(v,a,b).
   for(int other:incident[v]){if(contains(faces[other],u))continue;Face check=faces[other];std::sort(check.begin(),check.end());if(check==key)return reject("link_edge_duplicate");}
   Vec a=normal(old),b=normal(next);double nb=std::sqrt(dot(b,b));
   if(nb<=1e-14||dot(a,b)<=cos_limit*std::sqrt(dot(a,a))*nb)return reject("geometry");
  }
  if(policy!="conservative"){auto why=CollapsePolicy{faces,incident}.structure_check(u,v);if(!why.empty())return reject(why);}
  return true;
 }
 void rebuild(){decltype(heap) empty;heap.swap(empty);++rebuilds;
  for(int u=0;u<int(xyz.size());++u)if(alive_v[u])for(int v:neighbors(u))enqueue(u,v);}
 bool step(double error_limit=std::numeric_limits<double>::infinity()){
  size_t attempts=0;
  while(!heap.empty()){
   if(++attempts%10000==0&&PyErr_CheckSignals()!=0)throw py::error_already_set();
   if(policy=="error-adaptive"&&heap.top().cost>error_limit)return false;
   Item item=heap.top();heap.pop();int u=item.u,v=item.v;
   if(item.vu!=version[u]||item.vv!=version[v]||!legal(u,v))continue;
   if(policy=="error-adaptive"){
    double full=std::max(pending_radius*pending_pixel_scale,pending_appearance_radius*pending_app_scale);
    // Cheap queue score is a lower bound; orphan retirement may raise it.
    if(full>item.cost){heap.push({full,u,v,version[u],version[v]});continue;}
   }
   if(policy!="conservative"&&policy!="error-adaptive"){if(roles[u]==0)++accepted_interior;else {++accepted_chain;
    if(CollapsePolicy{faces,incident}.edge_count(u,v)==1)++accepted_boundary_chain;else ++accepted_nonmanifold_chain;}
    if(locked[v])++accepted_into_legacy_locked;}
   auto affected=neighbors(u);auto other=neighbors(v);affected.insert(affected.end(),other.begin(),other.end());affected.push_back(u);affected.push_back(v);unique(affected);
   auto changed=incident[u];std::sort(changed.begin(),changed.end());
   for(int fid:changed){for(int w:faces[fid]){auto &a=incident[w];a.erase(std::remove(a.begin(),a.end(),fid),a.end());}}
   for(int fid:changed){Face next=faces[fid];for(int &w:next)if(w==u)w=v;
    bool degenerate=next[0]==next[1]||next[1]==next[2]||next[0]==next[2];
    bool extra=policy=="error-adaptive"&&std::binary_search(pending_drop.begin(),pending_drop.end(),fid);
    if(degenerate||extra){alive_f[fid]=0;--count;if(extra&&!degenerate)++duplicate_faces_removed;}
    else{faces[fid]=next;for(int w:next)incident[w].push_back(fid);}}
   alive_v[u]=0;for(int j=0;j<10;++j)q[v][j]+=q[u][j];collapses.push_back({u,v});
   if(policy=="error-adaptive"){
    radius[v]=pending_radius;appearance_radius[v]=pending_appearance_radius;pixel_scale[v]=pending_pixel_scale;app_scale[v]=pending_app_scale;
    max_accepted_cost=std::max(max_accepted_cost,std::max(pending_radius*pending_pixel_scale,pending_appearance_radius*pending_app_scale));
    for(int w:pending_orphans){if(!incident[w].empty())throw std::runtime_error("Orphan bookkeeping mismatch");
     alive_v[w]=0;collapses.push_back({w,v});++orphan_vertices_merged;
     for(int j=0;j<10;++j)q[v][j]+=q[w][j];}
   }
   for(int a:affected)++version[a];
   if(policy!="conservative"&&policy!="error-adaptive"){CollapsePolicy checker{faces,incident};for(int a:affected)roles[a]=alive_v[a]?checker.classify(a):3;}
   std::vector<uint64_t> edges;for(int a:affected)if(alive_v[a])for(int b:neighbors(a))edges.push_back(edge(a,b));
   std::sort(edges.begin(),edges.end());edges.erase(std::unique(edges.begin(),edges.end()),edges.end());
   for(auto e:edges){enqueue(e>>32,uint32_t(e));enqueue(uint32_t(e),e>>32);}
   if(heap.size()>size_t(8)*std::max(1,count))rebuild();
   return true;
  }return false;
 }
 py::dict snapshot(){
  size_t nv=std::count(alive_v.begin(),alive_v.end(),uint8_t(1));
  py::array_t<int64_t> ids(nv),fids(count),f({py::ssize_t(count),py::ssize_t(3)});
  auto a=ids.mutable_unchecked<1>();auto b=fids.mutable_unchecked<1>();auto c=f.mutable_unchecked<2>();
  std::vector<int> rows(xyz.size(),-1);int row=0;
  for(int i=0;i<int(xyz.size());++i)if(alive_v[i]){rows[i]=row;a(row++)=i;}
  row=0;for(int i=0;i<int(faces.size());++i)if(alive_f[i]){b(row)=i;for(int j=0;j<3;++j)c(row,j)=rows[faces[i][j]];++row;}
  py::dict out;out["ids"]=ids;out["face_ids"]=fids;out["faces"]=f;return out;
 }
};
static py::dict build(py::array_t<double,py::array::c_style|py::array::forcecast> v,
 py::array_t<int64_t,py::array::c_style|py::array::forcecast> f,std::vector<double> ratios,bool strict,py::object progress,
 std::string policy="conservative",double max_angle=80.,bool area_weighted=false){
 if(policy=="error-adaptive")throw std::runtime_error("Use build_error for error-adaptive; ratios are not supported");
 NativeQEM mesh(v,f,progress,policy,max_angle,area_weighted);py::list levels;std::vector<int64_t> offsets={0};levels.append(mesh.snapshot());int initial=mesh.count;
 for(auto it=ratios.rbegin()+1;it!=ratios.rend();++it){int target=std::max(1,int(std::ceil(initial * *it)));int previous=mesh.count;
  while(mesh.count>target&&mesh.step()){if(mesh.collapses.size()%10000==0){if(PyErr_CheckSignals()!=0)throw py::error_already_set();if(!progress.is_none()){
    py::dict p;p["faces"]=mesh.count;p["collapses"]=mesh.collapses.size();p["heap_entries"]=mesh.heap.size();progress(p);}}}
  if(mesh.count>target&&!progress.is_none()){
   py::dict p;p["phase"]="budget unreachable";p["target_faces"]=target;p["actual_faces"]=mesh.count;p["rejections"]=mesh.rejected;progress(p);}
  if(mesh.count>target&&strict)throw std::runtime_error("Legal collapses exhausted before budget; use --pm-allow-incomplete to keep actual levels");
  if(mesh.count<previous){levels.append(mesh.snapshot());offsets.push_back(mesh.collapses.size());}
  if(mesh.count>target)break;
 }
 if(levels.size()<2)throw std::runtime_error("No nontrivial hierarchy possible under boundary/manifold safeguards");
 py::array_t<int64_t> collapses({py::ssize_t(mesh.collapses.size()),py::ssize_t(2)});auto c=collapses.mutable_unchecked<2>();
 for(size_t i=0;i<mesh.collapses.size();++i){c(i,0)=mesh.collapses[i][0];c(i,1)=mesh.collapses[i][1];}
 py::dict out;out["levels"]=levels;out["offsets"]=offsets;out["collapses"]=collapses;
 out["locked_vertices"]=std::count(mesh.locked.begin(),mesh.locked.end(),uint8_t(1));out["peak_heap_entries"]=mesh.peak_heap;out["heap_rebuilds"]=mesh.rebuilds;
 out["rejections"]=mesh.rejected;out["initial_role_counts"]=mesh.initial_roles;
 out["accepted_interior"]=mesh.accepted_interior;out["accepted_chain"]=mesh.accepted_chain;
 out["accepted_boundary_chain"]=mesh.accepted_boundary_chain;out["accepted_nonmanifold_chain"]=mesh.accepted_nonmanifold_chain;
 out["accepted_into_legacy_locked"]=mesh.accepted_into_legacy_locked;return out;
}
#include "topology_diagnostics.h"
static py::dict build_error(py::array_t<double,py::array::c_style|py::array::forcecast> v,
 py::array_t<int64_t,py::array::c_style|py::array::forcecast> f,
 py::array_t<double,py::array::c_style|py::array::forcecast> appearance,
 std::vector<double> limits,double max_angle,py::object progress,
 py::object pixel_scale,double appearance_unit,int64_t target_faces,py::object appearance_scale,int64_t target_vertices){
 if(limits.empty()&&target_faces<=0&&target_vertices<=0)throw std::runtime_error("Need increasing error levels, target_faces or target_vertices");
 if(target_vertices>0)target_faces=0; // v5: a vertex target takes precedence (each accepted merge removes exactly one vertex)
 if(!(appearance_unit>0)||!std::isfinite(appearance_unit))throw std::runtime_error("appearance_unit must be positive");
 double previous_limit=0.;for(double e:limits){if(!std::isfinite(e)||e<=previous_limit)throw std::runtime_error("Error levels must be finite, positive and increasing");previous_limit=e;}
 if(appearance.ndim()!=2||appearance.shape(0)!=v.shape(0)||appearance.shape(1)==0)throw std::runtime_error("Expected appearance [V,D]");
 NativeQEM mesh(v,f,progress,"error-adaptive",max_angle,true);
 auto a=appearance.unchecked<2>();mesh.appearance.resize(v.shape(0));
 for(int i=0;i<v.shape(0);++i){mesh.appearance[i].resize(appearance.shape(1));for(int j=0;j<appearance.shape(1);++j){
  if(!std::isfinite(a(i,j)))throw std::runtime_error("Nonfinite appearance");mesh.appearance[i][j]=a(i,j);}}
 mesh.appearance_unit=appearance_unit;std::fill(mesh.app_scale.begin(),mesh.app_scale.end(),appearance_unit);
 if(!appearance_scale.is_none()){
  auto as=py::cast<py::array_t<double,py::array::c_style|py::array::forcecast>>(appearance_scale);
  if(as.ndim()!=1||as.shape(0)!=v.shape(0))throw std::runtime_error("Expected appearance_scale [V] in px per appearance unit");
  auto r=as.unchecked<1>();
  for(int i=0;i<v.shape(0);++i){double x=r(i);if(!std::isfinite(x)||x<0)throw std::runtime_error("appearance_scale must be finite and >= 0");mesh.app_scale[i]=x;}
 }
 if(!pixel_scale.is_none()){
  auto ps=py::cast<py::array_t<double,py::array::c_style|py::array::forcecast>>(pixel_scale);
  if(ps.ndim()!=1||ps.shape(0)!=v.shape(0))throw std::runtime_error("Expected pixel_scale [V] in px per world unit");
  auto r=ps.unchecked<1>();
  for(int i=0;i<v.shape(0);++i){double x=r(i);if(!std::isfinite(x)||x<0)throw std::runtime_error("pixel_scale must be finite and >= 0");mesh.pixel_scale[i]=x*mesh.extent;}
 }
 mesh.rebuild();py::list levels;levels.append(mesh.snapshot());std::vector<int64_t> offsets={0};
 std::vector<double> actual_limits={0.};std::vector<double> skipped;
 if(target_vertices>0){
  int64_t alive0=0;for(uint8_t x:mesh.alive_v)alive0+=x;int steps=0;
  while(alive0-int64_t(mesh.collapses.size())>target_vertices&&mesh.step()){if(++steps%10000==0){
   if(PyErr_CheckSignals()!=0)throw py::error_already_set();
   if(!progress.is_none()){py::dict row;row["phase"]="cost-ordered merges to vertex target";row["target_vertices"]=target_vertices;
    row["vertices"]=alive0-int64_t(mesh.collapses.size());row["faces"]=mesh.count;row["vertex_merges"]=mesh.collapses.size();row["max_accepted_cost"]=mesh.max_accepted_cost;row["heap_entries"]=mesh.heap.size();progress(row);}
  }}
  if(alive0-int64_t(mesh.collapses.size())>target_vertices&&!progress.is_none()){py::dict row;row["phase"]="vertex target unreachable";row["vertices"]=alive0-int64_t(mesh.collapses.size());row["faces"]=mesh.count;row["rejections"]=mesh.rejected;progress(row);}
  levels.append(mesh.snapshot());offsets.push_back(mesh.collapses.size());actual_limits.push_back(mesh.max_accepted_cost);limits.clear();
 }
 if(target_faces>0){
  int steps=0;
  while(mesh.count>target_faces&&mesh.step()){if(++steps%10000==0){
   if(PyErr_CheckSignals()!=0)throw py::error_already_set();
   if(!progress.is_none()){py::dict row;row["phase"]="cost-ordered merges to face target";row["target_faces"]=target_faces;
    row["faces"]=mesh.count;row["vertex_merges"]=mesh.collapses.size();row["max_accepted_cost"]=mesh.max_accepted_cost;row["heap_entries"]=mesh.heap.size();progress(row);}
  }}
  if(mesh.count>target_faces&&!progress.is_none()){py::dict row;row["phase"]="face target unreachable";row["faces"]=mesh.count;row["rejections"]=mesh.rejected;progress(row);}
  levels.append(mesh.snapshot());offsets.push_back(mesh.collapses.size());actual_limits.push_back(mesh.max_accepted_cost);limits.clear();
 }
 for(double limit:limits){int previous=mesh.count;
  int steps=0;
  while(mesh.step(limit)){if(++steps%10000==0){
   if(PyErr_CheckSignals()!=0)throw py::error_already_set();
   if(!progress.is_none()){py::dict row;row["phase"]="error-based merges";row["error_limit"]=limit;
    row["faces"]=mesh.count;row["vertex_merges"]=mesh.collapses.size();row["heap_entries"]=mesh.heap.size();progress(row);}
  }}
  if(mesh.count<previous){levels.append(mesh.snapshot());offsets.push_back(mesh.collapses.size());actual_limits.push_back(limit);}else skipped.push_back(limit);
  if(!progress.is_none()){py::dict row;row["phase"]="error checkpoint";row["error_limit"]=limit;row["faces"]=mesh.count;row["vertex_merges"]=mesh.collapses.size();progress(row);}
 }
 if(levels.size()<2)throw std::runtime_error("No distinct error-based coarse level; inspect limits/geometry, no fallback used");
 py::array_t<int64_t> collapses({py::ssize_t(mesh.collapses.size()),py::ssize_t(2)});auto c=collapses.mutable_unchecked<2>();
 for(size_t i=0;i<mesh.collapses.size();++i){c(i,0)=mesh.collapses[i][0];c(i,1)=mesh.collapses[i][1];}
 py::dict out;out["levels"]=levels;out["offsets"]=offsets;out["collapses"]=collapses;
 out["error_limits"]=actual_limits;out["skipped_error_limits"]=skipped;out["rejections"]=mesh.rejected;
 out["median_edge_normalized"]=mesh.edge_unit;out["extent"]=mesh.extent;out["max_accepted_cost"]=mesh.max_accepted_cost;out["peak_heap_entries"]=mesh.peak_heap;
 out["duplicate_faces_removed"]=mesh.duplicate_faces_removed;out["orphan_vertices_merged"]=mesh.orphan_vertices_merged;
 return out;
}
PYBIND11_MODULE(_native_qem_vis,m){m.def("build",&build,py::arg("vertices"),py::arg("faces"),py::arg("ratios"),py::arg("strict"),py::arg("progress"),py::arg("policy")="conservative",py::arg("max_angle")=80.,py::arg("area_weighted")=false);m.def("diagnose_topology",&diagnose_topology);
 m.def("build_error",&build_error,py::arg("vertices"),py::arg("faces"),py::arg("appearance"),py::arg("limits"),py::arg("max_angle")=80.,py::arg("progress")=py::none(),
  py::arg("pixel_scale")=py::none(),py::arg("appearance_unit")=1.,py::arg("target_faces")=0,py::arg("appearance_scale")=py::none(),py::arg("target_vertices")=0);}

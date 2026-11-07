// Read-only combinatorial diagnostics. No geometry modification or edge collapse.
#include <map>
struct DiagnosticDSU {
 std::vector<int> p, rank;
 explicit DiagnosticDSU(int n):p(n),rank(n,0){for(int i=0;i<n;++i)p[i]=i;}
 int root(int a){while(p[a]!=a){p[a]=p[p[a]];a=p[a];}return a;}
 void join(int a,int b){a=root(a);b=root(b);if(a==b)return;if(rank[a]<rank[b])std::swap(a,b);p[b]=a;if(rank[a]==rank[b])++rank[a];}
};
static py::dict diagnose_topology(int64_t nv64,
 py::array_t<int64_t,py::array::c_style|py::array::forcecast> input,py::object progress){
 if(nv64<1||nv64>INT32_MAX||input.ndim()!=2||input.shape(1)!=3||input.shape(0)>INT32_MAX||input.shape(0)==0)
  throw std::runtime_error("Expected 1..INT32_MAX vertices and nonempty [F,3] faces");
 int nv=int(nv64),nf=int(input.shape(0));auto src=input.unchecked<2>();
 std::vector<Face> faces(nf),canonical;canonical.reserve(nf);
 std::vector<std::vector<int>> incident(nv);
 std::vector<uint64_t> edges;edges.reserve(size_t(nf)*3);
 std::vector<uint8_t> reasons(nv,0);DiagnosticDSU components(nv);
 for(int i=0;i<nf;++i){
  if(i%1000000==0){if(PyErr_CheckSignals()!=0)throw py::error_already_set();if(!progress.is_none()){py::dict row;row["phase"]="read incidence";row["faces_processed"]=i;progress(row);}}
  for(int j=0;j<3;++j){auto id=src(i,j);if(id<0||id>=nv)throw std::runtime_error("Face index out of range");faces[i][j]=int(id);}
  Face key=faces[i];std::sort(key.begin(),key.end());
  if(key[0]==key[1]||key[1]==key[2])throw std::runtime_error("Repeated-vertex triangle: diagnosis requires valid combinatorial triangles");
  canonical.push_back(key);
  for(int j=0;j<3;++j){int a=faces[i][j],b=faces[i][(j+1)%3];incident[a].push_back(i);edges.push_back((edge(a,b)<<1)|uint64_t(a>b));components.join(a,b);}
 }
 std::sort(canonical.begin(),canonical.end());int64_t duplicates=0;
 for(size_t i=1;i<canonical.size();++i)if(canonical[i]==canonical[i-1])++duplicates;
 std::vector<Face>().swap(canonical);
 if(!progress.is_none()){py::dict row;row["phase"]="sort edges and count incident faces";progress(row);}
 std::sort(edges.begin(),edges.end());std::map<int64_t,int64_t> edge_hist;
 int64_t inconsistent=0,unique_edges=0;
 for(size_t a=0;a<edges.size();){size_t b=a+1;uint64_t key=edges[a]>>1;while(b<edges.size()&&(edges[b]>>1)==key)++b;
  ++edge_hist[b-a];++unique_edges;int u=key>>32,v=uint32_t(key);
  if(b-a==1){reasons[u]|=1;reasons[v]|=1;}
  else if(b-a>2){reasons[u]|=2;reasons[v]|=2;}
  if(b-a==2&&(edges[a]&1)==(edges[a+1]&1))++inconsistent;
  a=b;
 }
 // A vertex's incident-face fans are the components of its link graph.
 // Unlike the builder, inspect ALSO vertices already locked by edge incidence.
 std::map<int64_t,int64_t> fan_hist,valence_hist;
 for(int u=0;u<nv;++u){
  if(u%250000==0){if(PyErr_CheckSignals()!=0)throw py::error_already_set();if(!progress.is_none()){py::dict row;row["phase"]="count all vertex fans";row["vertices_processed"]=u;progress(row);}}
  ++valence_hist[incident[u].size()];if(incident[u].empty())continue;
  std::vector<int> neighbors;neighbors.reserve(incident[u].size()*2);
  for(int fid:incident[u])for(int w:faces[fid])if(w!=u)neighbors.push_back(w);
  unique(neighbors);DiagnosticDSU link(neighbors.size());
  for(int fid:incident[u]){int a=-1,b=-1;for(int w:faces[fid])if(w!=u){if(a<0)a=w;else b=w;}
   link.join(std::lower_bound(neighbors.begin(),neighbors.end(),a)-neighbors.begin(),std::lower_bound(neighbors.begin(),neighbors.end(),b)-neighbors.begin());}
  int fans=0;for(int i=0;i<int(neighbors.size());++i)if(link.root(i)==i)++fans;
  ++fan_hist[fans];if(fans>1)reasons[u]|=4;
 }
 int64_t both_free=0,one_free=0,none_free=0;
 for(size_t a=0;a<edges.size();){size_t b=a+1;uint64_t key=edges[a]>>1;while(b<edges.size()&&(edges[b]>>1)==key)++b;
  if(b-a==2){int free=int(reasons[key>>32]==0)+int(reasons[uint32_t(key)]==0);if(free==2)++both_free;else if(free==1)++one_free;else ++none_free;}a=b;}
 int64_t masks[8]={0},unused=0;std::map<int,int64_t> component_sizes;
 std::array<std::vector<int>,8> sample_ids;
 for(int u=0;u<nv;++u){++masks[reasons[u]];if(incident[u].empty())++unused;else ++component_sizes[components.root(u)];
  if(sample_ids[reasons[u]].size()<12)sample_ids[reasons[u]].push_back(u);}
 std::vector<int64_t> sizes;for(auto item:component_sizes)sizes.push_back(item.second);std::sort(sizes.rbegin(),sizes.rend());if(sizes.size()>10)sizes.resize(10);
 auto hist=[](const std::map<int64_t,int64_t>& m){py::dict d;for(auto pair:m)d[py::str(std::to_string(pair.first))]=pair.second;return d;};
 py::dict combinations,samples;for(int i=0;i<8;++i){combinations[py::str(std::to_string(i))]=masks[i];samples[py::str(std::to_string(i))]=sample_ids[i];}
 py::dict out;out["vertices"]=nv;out["faces"]=nf;out["unique_edges"]=unique_edges;out["unused_vertices"]=unused;
 out["duplicate_faces_beyond_first"]=duplicates;out["edge_incidence_histogram"]=hist(edge_hist);
 out["two_face_edges_with_same_direction"]=inconsistent;out["vertex_incident_face_count_histogram"]=hist(valence_hist);
 out["vertex_fan_count_histogram_used_vertices"]=hist(fan_hist);out["reason_bitmask_counts"]=combinations;
 out["reason_bitmask_legend"]="1=boundary edge endpoint; 2=edge with >2 incident faces endpoint; 4=multiple face fans";
 out["boundary_vertices"]=masks[1]+masks[3]+masks[5]+masks[7];
 out["nonmanifold_edge_vertices"]=masks[2]+masks[3]+masks[6]+masks[7];
 out["multi_fan_vertices_all"]=masks[4]+masks[5]+masks[6]+masks[7];
 out["multi_fan_additional_locks"]=masks[4];out["builder_locked_union"]=nv-masks[0];
 out["two_face_edges_both_endpoints_unlocked"]=both_free;
 out["two_face_edges_one_endpoint_unlocked"]=one_free;out["two_face_edges_both_endpoints_locked"]=none_free;
 out["connected_components_used_vertices"]=component_sizes.size();out["largest_component_vertex_counts"]=sizes;
 out["example_vertex_ids_by_reason_bitmask"]=samples;return out;
}

// Experimental directed-collapse policies, independent of QEM ranking.
// Read-only local simulation. No automatic repair, position movement or welding.
#include <map>
#include <set>
#include <string>
struct PolicyDSU {
 std::vector<int> p;
 explicit PolicyDSU(int n):p(n){for(int i=0;i<n;++i)p[i]=i;}
 int root(int a){while(p[a]!=a){p[a]=p[p[a]];a=p[a];}return a;}
 void join(int a,int b){a=root(a);b=root(b);if(a!=b)p[b]=a;}
};
using TaggedFace=std::pair<int,Face>;
struct CollapsePolicy {
 const std::vector<Face>& faces;
 const std::vector<std::vector<int>>& incident;
 // Role: 0=regular interior, 1=uniform special-chain interior, 2=anchor, 3=unused.
 int edge_count(int a,int b)const{
  if(incident[a].size()>incident[b].size())std::swap(a,b);
  int n=0;for(int fid:incident[a])if(contains(faces[fid],b))++n;return n;
 }
 static std::vector<int> partition(const std::vector<TaggedFace>& local,int a,int b=-1){
  PolicyDSU d(local.size());std::map<uint64_t,int> owner;
  for(int i=0;i<int(local.size());++i){Face f=local[i].second;
   for(int j=0;j<3;++j){int u=f[j],v=f[(j+1)%3];
    if(u!=a&&v!=a&&u!=b&&v!=b)continue;
    auto inserted=owner.emplace(edge(u,v),i);if(!inserted.second)d.join(i,inserted.first->second);
   }}
  std::vector<int> result(local.size());for(int i=0;i<int(local.size());++i)result[i]=d.root(i);return result;
 }
 int classify(int u)const{
  if(incident[u].empty())return 3;
  std::map<int,int> counts;std::vector<TaggedFace> local;
  for(int fid:incident[u]){local.push_back({fid,faces[fid]});for(int w:faces[fid])if(w!=u)++counts[w];}
  auto parts=partition(local,u);unique(parts);if(parts.size()!=1)return 2;
  std::vector<int> labels;for(auto item:counts)if(item.second!=2)labels.push_back(item.second);
  if(labels.empty())return 0;
  if(labels.size()==2&&labels[0]==labels[1])return 1;
  return 2;
 }
 bool candidate(int u,int v,int role,const std::string& mode)const{
  if(role==0)return true;
  return mode=="stratified"&&role==1&&edge_count(u,v)!=2;
 }
 static bool preserves_partition(const std::vector<TaggedFace>& before,
                                 const std::vector<TaggedFace>& after,int anchor,int merged=-1){
  auto old=partition(before,anchor,merged),now=partition(after,anchor);
  std::map<int,int> fid_group,to_new,to_old;std::set<int> old_groups,represented;
  for(int i=0;i<int(before.size());++i){fid_group[before[i].first]=old[i];old_groups.insert(old[i]);}
  for(int i=0;i<int(after.size());++i){auto found=fid_group.find(after[i].first);if(found==fid_group.end())return false;
   int a=found->second,b=now[i];represented.insert(a);
   auto x=to_new.emplace(a,b),y=to_old.emplace(b,a);
   if((!x.second&&x.first->second!=b)||(!y.second&&y.first->second!=a))return false;
  }
  return represented==old_groups; // No fan disappears, splits or merges.
 }
 std::string structure_check(int u,int v)const{
  std::map<int,int> cu,cv;
  std::set<int> opposites;
  for(int fid:incident[u]){for(int w:faces[fid])if(w!=u)++cu[w];
   if(contains(faces[fid],v))for(int w:faces[fid])if(w!=u&&w!=v)opposites.insert(w);}
  for(int fid:incident[v])for(int w:faces[fid])if(w!=v)++cv[w];
  std::set<int> neighbors;for(auto e:cu)if(e.first!=v)neighbors.insert(e.first);for(auto e:cv)if(e.first!=u)neighbors.insert(e.first);
  // All changed edges after contraction are (v,w). Other edges retain incidence.
  for(int w:neighbors){int a=cu[w],b=cv[w],next=a+b-2*int(opposites.count(w));
   if(next<=0)return "purity"; // Mapped edge would be lost by triangle-only storage.
   int expected=0;
   if(a&&a!=2)expected=a;
   if(b&&b!=2){if(expected)return "stratum_merge";expected=b;}
   int actual=next==2?0:next;
   if(expected!=actual)return "stratum_label";
  }
  std::vector<int> affected(incident[u]);affected.insert(affected.end(),incident[v].begin(),incident[v].end());unique(affected);
  std::vector<TaggedFace> before,after;
  for(int fid:affected){Face f=faces[fid];before.push_back({fid,f});if(contains(f,u)&&contains(f,v))continue;
   for(int& w:f)if(w==u)w=v;after.push_back({fid,f});}
  if(!preserves_partition(before,after,v,u))return "fan_receiver";
  for(int w:neighbors){before.clear();after.clear();
   for(int fid:incident[w]){Face f=faces[fid];before.push_back({fid,f});if(contains(f,u)&&contains(f,v))continue;
    for(int& t:f)if(t==u)t=v;after.push_back({fid,f});}
   if(!preserves_partition(before,after,w))return "fan_neighbor";
  }
  return "";
 }
};

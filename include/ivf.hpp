#include "kmneas.hpp"

template <typename T>
class IVF_index {
  public:
    vector_dataset<T> centroids;
    vector<vector<int> > ivf;
    int NUM_CLUSTERS = 2;
    int PF_DIST = 4;

    IVF_index(const vector<int> &group,const vector<int> &membership,
        const vector_dataset<T> &_centroids) {

      centroids = _centroids;
      ivf.resize(centroids.num);
      for (int i = 0; i < group.size(); i++) ivf[membership[i]].push_back(group[i]);
    }
    IVF_index(const vector<int> &group,const string &filename) {
      vector<int> membership(group.size());
      std::ifstream in(filename + "-mem",std::ios::binary);
      in.read((char*) membership.data(),4*membership.size());
      in.close();
      centroids = vector_dataset<T>((filename + "-cen").c_str());
      ivf.resize(centroids.num);
      for (size_t i = 0; i < group.size(); i++) ivf[membership[i]].push_back(group[i]);
    }

    vector<int> sorted_near(vector_dataset<T> &data,T *query,int target = 15000) {
      vector<pair<float,int> > vv(ivf.size());
      for (int i = 0; i < ivf.size(); i++) {
        vv[i] = make_pair(compute_distance_squared(data.dim,centroids[i],query),i);
      }
      sort(vv.begin(),vv.end());
      vector<int> ret;
      for (int i = 0; i < vv.size(); i++) {
        ret.insert(ret.end(),ivf[vv[i].second].begin(),ivf[vv[i].second].end());
        if (ret.size() > target) break;
      }
      sort(ret.begin(),ret.end());
      return ret;
    }

    vector<int> sorted_near_filter(vector_dataset<T> &data,T *query,
        function<bool(int)> filter,int target = 10) {

      //int l = data.dim*sizeof(T)/64;
      vector<pair<float,int> > vv(ivf.size());
      for (size_t i = 0; i < ivf.size(); i++) {
        vv[i] = make_pair(compute_distance_squared(data.dim,centroids[i],query),i);
        // This access pattern is predictable
#ifdef USE_PF_VEC
        //if (i+PF_DIST < ivf.size()) {
        //    char *a = (char*) centroids[i+PF_DIST];
        //    for (int j = 0; j < l; j++) __builtin_prefetch(a+64*j);
        //}
#endif
      }
      partial_sort(vv.begin(),vv.begin()+min(NUM_CLUSTERS,(int) vv.size()),vv.end());
      vector<int> ret;
      for (int i = 0; i < min(NUM_CLUSTERS,(int) vv.size()); i++) {
        for (int x: ivf[vv[i].second]) {
          if (filter(x)) ret.push_back(x);
        }
        //if (ret.size() > target) break;
      }
      return ret;
    }
};

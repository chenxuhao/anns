#include <omp.h>
#include "graph.hpp"
#include "utils.hpp"
#include "common.hpp"
#include "pqueue.hpp"

#define USE_PQ

// Build kNN graph for given k
void build_knn_graph(int k, size_t N, size_t dim, float* data_points, Graph &g) {
  Timer t;
  t.Start();
  g.allocateFrom(N, N*k);
  #pragma omp parallel
  {
#ifdef USE_PQ
    pqueue_t<vid_t> queue(k); // priority queue
#else
    std::vector<std::pair<float, vid_t>> queue;
    queue.reserve(N);
#endif
    int progress = 0;
    #pragma omp for
    for (size_t i = 0; i < N; ++i) {
      progress++;
      if ((progress % 10) == 0) printf("%d\n",progress);
      queue.clear();
      // compute pair-wise distances
      for (size_t j = 0; j < N; ++j) {
        if (i != j) {
          auto dist = utils::compute_distance(dim, &data_points[i*dim], &data_points[j*dim]);
#ifdef USE_PQ
          if (dist < queue.get_tail_dist())
          queue.push(j, dist);
#else
          queue.emplace_back(dist, j);
#endif
        }
      }

      // Sort distances and keep only the top k neighbors
#ifndef USE_PQ
      std::sort(queue.begin(), queue.end());
#endif

      // Store indices of kNN neighbors for the current point i
      auto offset = i * k;
      g.fixEndEdge(i, offset + k);
      for (int j = 0; j < k; ++j) {
#ifdef USE_PQ
        g.constructEdge(offset+j, queue[j]);
#else
        g.constructEdge(offset+j, queue[j].second);
#endif
      }
    }
  }
  t.Stop();
#ifdef USE_PQ
  std::string name = "priority-queue";
#else
  std::string name = "vector-sort";
#endif
  printf("building kNN graph using %s time: %f sec\n", name.c_str(), t.Seconds());
}

// NN-descent algorithm
void build_approx_knn_graph(int k, size_t N, size_t dim, float* data_points, Graph &g,
  double rho = 1.0, double delta = 0.001) {
  Timer t;
  t.Start();
  g.allocateFrom(N, N*k);
  auto distance = [&](int i,int j) {
    return utils::compute_distance(dim, &data_points[i*dim], &data_points[j*dim]);
  };

  vector<pqueue_t<vid_t> > B(N,pqueue_t<vid_t>(k));
  // sample K random edges for each node
  #pragma omp parallel
  {
    mt19937 gen(123*omp_get_thread_num());
    #pragma omp for
    for (size_t i = 0; i < N; ++i) {
      while (B[i].size() < k) {
        size_t v = gen() % N;
        if (i != v) B[i].push(v,distance(i,v));
      }
    }
  }

  int num_iters = 0;
  size_t rhoK = (int) rho*k;
  vector<omp_lock_t> locks(N);
  for (size_t i = 0; i < N; i++) omp_init_lock(&locks[i]);
  ofstream debug("debug.txt");
  while (num_iters < 500) {
    Timer iter_timer;
    Timer iter_timer1;
    Timer iter_timer2;
    iter_timer.Start();
    iter_timer1.Start();
    iter_timer2.Start();
    printf("num_iters: %d\n",num_iters++);
    debug << "num_iters: " << num_iters << endl;
    // note: is_expanded is the opposite as in the original paper
    vector<vector<vid_t> > old_v(N);
    vector<vector<vid_t> > new_v(N);
    double sum = 0;
    #pragma omp parallel for reduction(+:sum)
    for (size_t i = 0; i < N; ++i) {
      for (int j = 0; j < k; j++) {
        sum += B[i].get_dist(j);
        if (B[i].is_expanded(j)) old_v[i].push_back(B[i][j]);
        else new_v[i].push_back(j);
      }
      if (new_v[i].size() > rhoK) {
        random_shuffle(new_v[i].begin(),new_v[i].end());
        new_v[i].resize(rhoK);
      }
      for (auto j: new_v[i]) B[i].set_expanded(j);
      for (auto &j: new_v[i]) j = B[i][j];
    }
    printf("sum: %f\n",sum);
    debug << "sum: " << sum << endl;
    iter_timer1.Stop();
    debug << "Time: " << iter_timer1.Seconds() << " sec" << endl;
    printf("finished step 1\n");
    vector<vector<vid_t> > old_r(N);
    vector<vector<vid_t> > new_r(N);
    for (size_t i = 0; i < N; ++i) {
      for (auto v: old_v[i]) old_r[v].push_back(i);
      for (auto v: new_v[i]) new_r[v].push_back(i);
    }
    iter_timer2.Stop();
    debug << "Time: " << iter_timer2.Seconds() << " sec" << endl;
    printf("finished step 2\n");

    size_t c = 0;
    #pragma omp parallel for reduction(+:c)
    for (size_t i = 0; i < N; ++i) {
      if ((i % 10000) == 0) printf("i: %lu\n",i);
      if (old_r[i].size() > rhoK) {
        random_shuffle(old_r[i].begin(),old_r[i].end());
        old_r[i].resize(rhoK);
      }
      old_v[i].insert(old_v[i].end(),old_r[i].begin(),old_r[i].end());
      if (new_r[i].size() > rhoK) {
        random_shuffle(new_r[i].begin(),new_r[i].end());
        new_r[i].resize(rhoK);
      }
      new_v[i].insert(new_v[i].end(),new_r[i].begin(),new_r[i].end());
      for (size_t a = 0; a < new_v[i].size(); a++) {
        for (size_t b = a+1; b < new_v[i].size(); b++) {
          auto u1 = new_v[i][a];
          auto u2 = new_v[i][b];
          if (u1 == u2) continue;
          auto l = distance(u1,u2);
          if (l < B[u1].get_tail_dist()) {
            omp_set_lock(&locks[u1]);
            c += (B[u1].push(u2,l) >= 0);
            omp_unset_lock(&locks[u1]);
          }
          if (l < B[u2].get_tail_dist()) {
            omp_set_lock(&locks[u2]);
            c += (B[u2].push(u1,l) >= 0);
            omp_unset_lock(&locks[u2]);
          }
        }
      }
      for (size_t a = 0; a < new_v[i].size(); a++) {
        for (size_t b = 0; b < old_v[i].size(); b++) {
          auto u1 = new_v[i][a];
          auto u2 = old_v[i][b];
          if (u1 == u2) continue;
          auto l = distance(u1,u2);
          if (l < B[u1].get_tail_dist()) {
            omp_set_lock(&locks[u1]);
            c += (B[u1].push(u2,l) >= 0);
            omp_unset_lock(&locks[u1]);
          }
          if (l < B[u2].get_tail_dist()) {
            omp_set_lock(&locks[u2]);
            c += (B[u2].push(u1,l) >= 0);
            omp_unset_lock(&locks[u2]);
          }
        }
      }
    }

    iter_timer.Stop();
    printf("Time: %f sec\n",iter_timer.Seconds());
    debug << "Time: " << iter_timer.Seconds() << " sec" << endl;
    printf("c: %lu\n",c);
    debug << "c: " << c << endl;
    if (c < delta*N*k) break;
  }
  for (size_t i = 0; i < N; i++) omp_destroy_lock(&locks[i]);

  // construct the graph
  #pragma omp parallel for
  for (size_t i = 0; i < N; ++i) {
    auto offset = i * k;
    g.fixEndEdge(i, offset + k);
    for (int j = 0; j < k; ++j) g.constructEdge(offset+j, B[i][j]);
  }

  t.Stop();
  printf("building approx kNN graph using NN-descent time: %f sec\n", t.Seconds());
}

void pruning(Graph &g) {
}


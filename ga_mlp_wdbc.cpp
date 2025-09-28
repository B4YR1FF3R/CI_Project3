// file: ga_mlp_wdbc.cpp
// compile: g++ -std=c++17 -O2 ga_mlp_wdbc.cpp -o ga_mlp_wdbc
#include <bits/stdc++.h>
using namespace std;

/* ---------- utility ---------- */
static double sigmoid(double x){ return 1.0/(1.0+exp(-x)); }
static double randd(){ return (double)rand()/RAND_MAX; }
static double gaussian_noise(double sigma){
    // Box-Muller
    double u1 = randd(), u2 = randd();
    if(u1<1e-12) u1=1e-12;
    return sigma * sqrt(-2*log(u1)) * cos(2*acos(-1.0)*u2);
}

/* ---------- Dataset ---------- */
struct Sample {
    vector<double> x; // size 30
    int y; // 0 or 1
};
vector<Sample> load_wdbc(const string &path){
    ifstream in(path);
    if(!in) throw runtime_error("Cannot open dataset file: " + path);
    vector<Sample> data;
    string line;
    while (getline(in,line)){
        if(line.size()<5) continue;
        stringstream ss(line);
        string tok;
        // ID
        getline(ss, tok, ','); 
        // label
        getline(ss, tok, ',');
        int y = (tok=="M")?1:0;
        vector<double> feats;
        while(getline(ss, tok, ',')){
            feats.push_back(stod(tok));
        }
        if(feats.size()>=30) {
            feats.resize(30);
            data.push_back({feats,y});
        }
    }
    return data;
}

/* ---------- Normalization (z-score) ---------- */
struct Normalizer {
    vector<double> mean, std;
    void fit(const vector<Sample>& data){
        int nfeat = data[0].x.size();
        mean.assign(nfeat,0.0);
        std.assign(nfeat,0.0);
        for(auto &s: data) for(int i=0;i<nfeat;i++) mean[i]+=s.x[i];
        for(int i=0;i<nfeat;i++) mean[i]/=data.size();
        for(auto &s: data) for(int i=0;i<nfeat;i++){
            double d = s.x[i]-mean[i];
            std[i] += d*d;
        }
        for(int i=0;i<nfeat;i++){
            std[i] = sqrt(std[i]/(data.size()-1));
            if(std[i] < 1e-8) std[i] = 1.0; // avoid div0
        }
    }
    vector<double> transform(const vector<double>& x) const {
        vector<double> r(x.size());
        for(size_t i=0;i<x.size();i++) r[i] = (x[i]-mean[i])/std[i];
        return r;
    }
};

/* ---------- MLP representation built from weight vector ---------- */
struct MLP {
    vector<int> layers; // e.g. [30, h1, h2, 1]
    vector<vector<vector<double>>> W; // W[l][i][j] weight from node j in layer l to node i in layer l+1
    vector<vector<double>> B; // biases per layer l->l+1
    MLP(const vector<int>& layers_): layers(layers_){
        int L = layers.size()-1;
        W.resize(L);
        B.resize(L);
        for(int l=0;l<L;l++){
            W[l].assign(layers[l+1], vector<double>(layers[l],0.0));
            B[l].assign(layers[l+1], 0.0);
        }
    }
    // create from flat vector
    static MLP from_vector(const vector<int>& layers, const vector<double>& flat){
        MLP m(layers);
        size_t idx=0;
        for(int l=0;l< (int)layers.size()-1; ++l){
            for(int i=0;i<layers[l+1]; ++i){
                for(int j=0;j<layers[l]; ++j){
                    m.W[l][i][j] = flat[idx++]; 
                }
            }
            for(int i=0;i<layers[l+1]; ++i) m.B[l][i] = flat[idx++];
        }
        if(idx != flat.size()){
            cerr<<"Warning: vector size mismatch in from_vector: idx="<<idx<<" flat="<<flat.size()<<"\n";
        }
        return m;
    }
    static size_t required_vector_size(const vector<int>& layers){
        size_t s=0;
        for(int l=0;l< (int)layers.size()-1; ++l){
            s += (size_t)layers[l+1]*layers[l]; // weights
            s += (size_t)layers[l+1]; // biases
        }
        return s;
    }
    double predict_prob(const vector<double>& x) const {
        vector<double> cur = x;
        for(int l=0; l< (int)layers.size()-1; ++l){
            vector<double> next(layers[l+1],0.0);
            for(int i=0;i<layers[l+1];++i){
                double z = B[l][i];
                for(int j=0;j<layers[l];++j) z += W[l][i][j]*cur[j];
                // hidden and output both use sigmoid (for binary)
                next[i] = sigmoid(z);
            }
            cur.swap(next);
        }
        return cur[0]; // final output node (probability)
    }
    int predict(const vector<double>& x, double thresh=0.5) const {
        return predict_prob(x) >= thresh ? 1 : 0;
    }
};

/* ---------- GA Chromosome helpers ---------- */
struct Individual {
    vector<double> genes;
    double fitness;
    Individual() : fitness(0.0) {}
};
bool cmp_ind(const Individual &a, const Individual &b){ return a.fitness > b.fitness; }

/* ---------- Fitness evaluation: given individual genes -> build MLP -> CV accuracy ---------- */
double evaluate_individual(const Individual &ind, const vector<int>& layers,
                           const vector<vector<Sample>>& folds, bool verbose=false)
{
    size_t req = MLP::required_vector_size(layers);
    if(ind.genes.size() != req) {
        cerr<<"Gene size mismatch. expected "<<req<<" got "<<ind.genes.size()<<"\n";
        return 0.0;
    }
    // For each fold: train? (with GA we DON'T train weights; weights are the genes)
    // But we must evaluate on folds where normalization is fit on training portion.
    // Approach: for each fold, combine other folds as train_normer fit, then transform test -> evaluate.
    int K = folds.size();
    double sumacc = 0.0;
    for(int k=0;k<K;k++){
        // compose training set
        vector<Sample> trainset, testset;
        for(int i=0;i<K;i++){
            if(i==k) testset.insert(testset.end(), folds[i].begin(), folds[i].end());
            else trainset.insert(trainset.end(), folds[i].begin(), folds[i].end());
        }
        Normalizer norm;
        norm.fit(trainset);
        // build normalized test vectors
        vector<pair<vector<double>,int>> testdata;
        for(auto &s: testset){
            testdata.push_back({norm.transform(s.x), s.y});
        }
        // Build MLP from genes
        MLP mlp = MLP::from_vector(layers, ind.genes);
        // evaluate accuracy on testdata
        int correct=0;
        for(auto &p: testdata){
            int pred = mlp.predict(p.first);
            if(pred == p.second) correct++;
        }
        double acc = (testdata.empty()?0.0: (double)correct / testdata.size());
        sumacc += acc;
    }
    return sumacc / K;
}

/* ---------- GA operators ---------- */
Individual tournament_select(const vector<Individual>& pop, int tsize){
    int n = pop.size();
    Individual best; best.fitness = -1e9;
    for(int i=0;i<tsize;i++){
        int r = rand()%n;
        if(pop[r].fitness > best.fitness) best = pop[r];
    }
    return best;
}
pair<Individual,Individual> crossover_blend(const Individual &a, const Individual &b, double alpha){
    Individual c1=a, c2=b;
    int L = a.genes.size();
    for(int i=0;i<L;i++){
        double u = randd();
        double g1 = alpha * a.genes[i] + (1-alpha) * b.genes[i];
        double g2 = alpha * b.genes[i] + (1-alpha) * a.genes[i];
        c1.genes[i] = g1;
        c2.genes[i] = g2;
    }
    return {c1,c2};
}
void mutate(Individual &ind, double mrate, double sigma){
    int L = ind.genes.size();
    for(int i=0;i<L;i++){
        if(randd() < mrate){
            ind.genes[i] += gaussian_noise(sigma);
        }
    }
}

/* ---------- Build K folds (stratified simple) ---------- */
vector<vector<Sample>> build_folds(const vector<Sample>& data, int K){
    // simple stratified by label
    vector<Sample> pos, neg;
    for(auto &s: data) (s.y==1?pos:neg).push_back(s);
    vector<vector<Sample>> folds(K);
    for(size_t i=0;i<pos.size();++i) folds[i%K].push_back(pos[i]);
    for(size_t i=0;i<neg.size();++i) folds[i%K].push_back(neg[i]);
    return folds;
}

/* ---------- Main GA training loop (evaluating with CV) ---------- */
Individual run_GA(const vector<int>& layers, vector<vector<Sample>>& folds,
                  int pop_size=100, int generations=200, double crossover_rate=0.7,
                  double mutation_rate=0.01, double mutation_sigma=0.1, int tour_size=3)
{
    size_t genesz = MLP::required_vector_size(layers);
    // initialize population
    vector<Individual> pop(pop_size);
    for(int i=0;i<pop_size;i++){
        pop[i].genes.resize(genesz);
        for(size_t g=0; g<genesz; ++g){
            pop[i].genes[g] = gaussian_noise(1.0); // random init
        }
    }
    // evaluate initial
    for(auto &ind: pop){
        ind.fitness = evaluate_individual(ind, layers, folds);
    }
    for(int gen=0; gen<generations; ++gen){
        vector<Individual> newpop;
        // elitism: keep best 2
        sort(pop.begin(), pop.end(), cmp_ind);
        newpop.push_back(pop[0]); if(pop_size>1) newpop.push_back(pop[1]);
        while((int)newpop.size() < pop_size){
            Individual p1 = tournament_select(pop, tour_size);
            Individual p2 = tournament_select(pop, tour_size);
            Individual c1 = p1, c2 = p2;
            if(randd() < crossover_rate){
                auto pr = crossover_blend(p1,p2,0.5);
                c1 = pr.first; c2 = pr.second;
            }
            mutate(c1, mutation_rate, mutation_sigma);
            mutate(c2, mutation_rate, mutation_sigma);
            c1.fitness = evaluate_individual(c1, layers, folds);
            if((int)newpop.size() < pop_size) newpop.push_back(c1);
            if((int)newpop.size() < pop_size){
                c2.fitness = evaluate_individual(c2, layers, folds);
                newpop.push_back(c2);
            }
        }
        pop.swap(newpop);
        // optional logging
        if(gen%10==0){
            sort(pop.begin(), pop.end(), cmp_ind);
            cerr<<"Gen "<<gen<<" best fitness="<<pop[0].fitness<<"\n";
        }
    }
    sort(pop.begin(), pop.end(), cmp_ind);
    return pop[0];
}

/* ---------- main ---------- */
int main(int argc, char** argv){
    srand(12345);
    if(argc < 2){
        cerr<<"Usage: "<<argv[0]<<" /path/to/wdbc.txt\n";
        return 1;
    }
    string path = argv[1];
    auto data = load_wdbc(path);
    cerr<<"Loaded samples: "<<data.size()<<"\n";
    // build 10 folds
    int K = 10; // change to 10%-holdout if needed (see comment below)
    auto folds = build_folds(data, K);

    // Example: try architectures by changing hidden layers and nodes
    // Example architectures to try:
    vector<vector<int>> architectures = {
        // input 30 -> hidden -> output 1
        {30, 8, 1},
        {30, 16, 1},
        {30, 16, 8, 1},
        {30, 32, 16, 1}
    };

    for(auto &arch : architectures){
        cerr<<"Running GA for arch: ";
        for(int x: arch) cerr<<x<<" ";
        cerr<<"\n";
        Individual best = run_GA(arch, folds,
                                 60, // population
                                 120, // generations
                                 0.8, // crossover rate
                                 0.02, // mutation rate
                                 0.1, // mutation sigma
                                 3); // tournament
        cerr<<"Best fitness (mean CV accuracy): "<<best.fitness<<"\n";
    }

    return 0;
}

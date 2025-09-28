// ตัว demo จะเป็นตัวภาษาไทย ไว้ใช้ดูเฉยๆ
// ไฟล์ ga_mle_wdbe_demo (ที่ใช้ mle เพราะติด e=example ไว้ล้อชื่อกับ mlp เฉยๆ)
// โปรแกรมตัวอย่าง: การฝึก Multilayer Perceptron (MLP) ด้วย Genetic Algorithm (GA)
// ใช้ข้อมูล wdbc.txt (Breast Cancer Wisconsin Diagnostic) เพื่อทดลองการจำแนก (classification)
// ก่อนรันคอมไพล์ก่อน: g++ -std=c++17 -O2 ga_mle_wdbc_demo.cpp -o ga_mle_wdbc_demo
// แล้วพิมพ์อันนี้ลงในเทอร์มินอล: chcp 65001
// รัน: .\ga_mle_wdbc_demo.exe

#include <bits/stdc++.h>
using namespace std;

// ---------- ฟังก์ชันพื้นฐาน ----------

// sigmoid: ฟังก์ชันทางคณิตศาสตร์ที่ใช้ใน neuron เพื่อตัดสินผลลัพธ์
static double sigmoid(double x){ return 1.0/(1.0+exp(-x)); }

// randd: สุ่มค่าแบบทศนิยมระหว่าง 0 ถึง 1
static double randd(){ return (double)rand()/RAND_MAX; }

// gaussian_noise: สร้าง noise แบบสุ่มตามการแจกแจงปกติ (Normal Distribution)
// ใช้ Box-Muller Transform
static double gaussian_noise(double sigma){
    double u1 = randd(), u2 = randd();
    if(u1 < 1e-12) u1 = 1e-12;
    return sigma * sqrt(-2*log(u1)) * cos(2*acos(-1.0)*u2);
}

// ---------- โครงสร้างข้อมูล ----------

// Sample: เก็บข้อมูลแต่ละตัวอย่าง (input 30 ค่า, label = 0 หรือ 1)
struct Sample {
    vector<double> x; // input features
    int y;            // label (0=benign, 1=malignant)
};

// โหลดข้อมูลจากไฟล์ wdbc.txt
vector<Sample> load_wdbc(const string &path){
    ifstream in(path);
    if(!in) throw runtime_error("ไม่สามารถเปิดไฟล์ dataset ได้: " + path);
    vector<Sample> data;
    string line;
    while (getline(in,line)){
        if(line.size()<5) continue;
        stringstream ss(line);
        string tok;
        // คอลัมน์แรกเป็น ID (ข้ามไป)
        getline(ss, tok, ',');
        // คอลัมน์ที่สองเป็น label (M/B)
        getline(ss, tok, ',');
        int y = (tok=="M")?1:0;
        // อ่านค่า features ทั้ง 30 ค่า
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

// ---------- Normalization ----------
// ใช้ทำให้ข้อมูล input มีค่าใกล้เคียงกัน (ค่าเฉลี่ย = 0, ส่วนเบี่ยงเบนมาตรฐาน = 1)
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
            if(std[i] < 1e-8) std[i] = 1.0; // ป้องกันการหารด้วยศูนย์
        }
    }
    vector<double> transform(const vector<double>& x) const {
        vector<double> r(x.size());
        for(size_t i=0;i<x.size();i++) r[i] = (x[i]-mean[i])/std[i];
        return r;
    }
};

// ---------- โครงสร้าง MLP ----------
// layers = ขนาดของแต่ละชั้น (input, hidden, output)
// weights และ biases จะถูกเก็บในรูปแบบ vector
struct MLP {
    vector<int> layers; 
    vector<vector<vector<double>>> W; 
    vector<vector<double>> B; 
    MLP(const vector<int>& layers_): layers(layers_){
        int L = layers.size()-1;
        W.resize(L);
        B.resize(L);
        for(int l=0;l<L;l++){
            W[l].assign(layers[l+1], vector<double>(layers[l],0.0));
            B[l].assign(layers[l+1], 0.0);
        }
    }
    // แปลงจาก chromosome (genes) ไปเป็นโครงสร้าง MLP
    static MLP from_vector(const vector<int>& layers, const vector<double>& flat){
        MLP m(layers);
        size_t idx=0;
        for(int l=0;l<(int)layers.size()-1; ++l){
            for(int i=0;i<layers[l+1]; ++i){
                for(int j=0;j<layers[l]; ++j){
                    m.W[l][i][j] = flat[idx++]; 
                }
            }
            for(int i=0;i<layers[l+1]; ++i) m.B[l][i] = flat[idx++];
        }
        return m;
    }
    // คำนวณขนาดของ vector weights+biases ทั้งหมด
    static size_t required_vector_size(const vector<int>& layers){
        size_t s=0;
        for(int l=0;l<(int)layers.size()-1; ++l){
            s += (size_t)layers[l+1]*layers[l];
            s += (size_t)layers[l+1];
        }
        return s;
    }
    // ทำ forward pass
    double predict_prob(const vector<double>& x) const {
        vector<double> cur = x;
        for(int l=0; l<(int)layers.size()-1; ++l){
            vector<double> next(layers[l+1],0.0);
            for(int i=0;i<layers[l+1];++i){
                double z = B[l][i];
                for(int j=0;j<layers[l];++j) z += W[l][i][j]*cur[j];
                next[i] = sigmoid(z);
            }
            cur.swap(next);
        }
        return cur[0];
    }
    int predict(const vector<double>& x, double thresh=0.5) const {
        return predict_prob(x) >= thresh ? 1 : 0;
    }
};

// ---------- โครงสร้าง Individual (สมาชิกของ GA) ----------
struct Individual {
    vector<double> genes; // weights+biases
    double fitness;       // ความแม่นยำเฉลี่ยจาก cross validation
    Individual() : fitness(0.0) {}
};
bool cmp_ind(const Individual &a, const Individual &b){ return a.fitness > b.fitness; }

// ---------- การประเมิน fitness ของแต่ละ individual ----------
double evaluate_individual(const Individual &ind, const vector<int>& layers,
                           const vector<vector<Sample>>& folds)
{
    size_t req = MLP::required_vector_size(layers);
    if(ind.genes.size() != req) return 0.0;
    int K = folds.size();
    double sumacc = 0.0;
    for(int k=0;k<K;k++){
        vector<Sample> trainset, testset;
        for(int i=0;i<K;i++){
            if(i==k) testset.insert(testset.end(), folds[i].begin(), folds[i].end());
            else trainset.insert(trainset.end(), folds[i].begin(), folds[i].end());
        }
        Normalizer norm;
        norm.fit(trainset);
        vector<pair<vector<double>,int>> testdata;
        for(auto &s: testset){
            testdata.push_back({norm.transform(s.x), s.y});
        }
        MLP mlp = MLP::from_vector(layers, ind.genes);
        int correct=0;
        for(auto &p: testdata){
            int pred = mlp.predict(p.first);
            if(pred == p.second) correct++;
        }
        double acc = (double)correct / testdata.size();
        sumacc += acc;
    }
    return sumacc / K;
}

// ---------- ฟังก์ชันของ GA ----------

// tournament selection
Individual tournament_select(const vector<Individual>& pop, int tsize){
    int n = pop.size();
    Individual best; best.fitness = -1e9;
    for(int i=0;i<tsize;i++){
        int r = rand()%n;
        if(pop[r].fitness > best.fitness) best = pop[r];
    }
    return best;
}

// crossover (blend)
pair<Individual,Individual> crossover_blend(const Individual &a, const Individual &b, double alpha){
    Individual c1=a, c2=b;
    int L = a.genes.size();
    for(int i=0;i<L;i++){
        double g1 = alpha * a.genes[i] + (1-alpha) * b.genes[i];
        double g2 = alpha * b.genes[i] + (1-alpha) * a.genes[i];
        c1.genes[i] = g1;
        c2.genes[i] = g2;
    }
    return {c1,c2};
}

// mutation (สุ่มปรับค่าเล็กน้อย)
void mutate(Individual &ind, double mrate, double sigma){
    int L = ind.genes.size();
    for(int i=0;i<L;i++){
        if(randd() < mrate){
            ind.genes[i] += gaussian_noise(sigma);
        }
    }
}

// ---------- สร้าง folds สำหรับ cross validation ----------
vector<vector<Sample>> build_folds(const vector<Sample>& data, int K){
    vector<Sample> pos, neg;
    for(auto &s: data) (s.y==1?pos:neg).push_back(s);
    vector<vector<Sample>> folds(K);
    for(size_t i=0;i<pos.size();++i) folds[i%K].push_back(pos[i]);
    for(size_t i=0;i<neg.size();++i) folds[i%K].push_back(neg[i]);
    return folds;
}

// ---------- GA training loop ----------
Individual run_GA(const vector<int>& layers, vector<vector<Sample>>& folds,
                  int pop_size=60, int generations=100, double crossover_rate=0.7,
                  double mutation_rate=0.01, double mutation_sigma=0.1, int tour_size=3)
{
    size_t genesz = MLP::required_vector_size(layers);
    vector<Individual> pop(pop_size);
    for(int i=0;i<pop_size;i++){
        pop[i].genes.resize(genesz);
        for(size_t g=0; g<genesz; ++g){
            pop[i].genes[g] = gaussian_noise(1.0);
        }
    }
    for(auto &ind: pop){
        ind.fitness = evaluate_individual(ind, layers, folds);
    }
    for(int gen=0; gen<generations; ++gen){
        vector<Individual> newpop;
        sort(pop.begin(), pop.end(), cmp_ind);
        newpop.push_back(pop[0]);
        if(pop_size>1) newpop.push_back(pop[1]);
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
        if(gen%10==0){
            sort(pop.begin(), pop.end(), cmp_ind);
            cerr<<"รุ่น "<<gen<<" fitness ที่ดีที่สุด="<<pop[0].fitness<<"\n";
        }
    }
    sort(pop.begin(), pop.end(), cmp_ind);
    return pop[0];
}

// ---------- main program ----------
int main(){
    srand(12345);
    string path = "wdbc.txt"; // ใช้ไฟล์ wdbc.txt โดยอัตโนมัติ
    auto data = load_wdbc(path);
    cerr<<"โหลดข้อมูลสำเร็จ จำนวนตัวอย่างทั้งหมด: "<<data.size()<<"\n";

    int K = 10; 
    auto folds = build_folds(data, K);

    // กำหนดโครงสร้าง network ที่จะทดลอง
    vector<vector<int>> architectures = {
        {30, 8, 1},
        {30, 16, 1},
        {30, 16, 8, 1},
        {30, 32, 16, 1}
    };

    for(auto &arch : architectures){
        cerr<<"กำลังรัน GA สำหรับสถาปัตยกรรม: ";
        for(int x: arch) cerr<<x<<" ";
        cerr<<"\n";
        Individual best = run_GA(arch, folds,
                                 60,   // ขนาดประชากร
                                 100,  // จำนวน generation
                                 0.8,  // อัตรา crossover
                                 0.02, // อัตรา mutation
                                 0.1,  // sigma ของ mutation
                                 3);   // tournament size
        cerr<<"ผลลัพธ์ที่ดีที่สุด (accuracy เฉลี่ย 10-fold CV): "<<best.fitness<<"\n";
    }
    return 0;
}

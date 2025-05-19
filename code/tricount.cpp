#include <bits/stdc++.h>
#include <vector>
#include <iostream>
#include <algorithm>
#include <string>
#include <numeric>
#include <sstream>
#include <fstream>
#include <queue>
#include <climits>
#include <Kokkos_Core.hpp>

using namespace std;

// Define MemSpace based on the enabled Kokkos backend
#ifdef KOKKOS_ENABLE_CUDA
    #define MemSpace Kokkos::CudaSpace
#else
    #define MemSpace Kokkos::HostSpace
#endif

using ExecSpace = MemSpace::execution_space;
using range_policy = Kokkos::RangePolicy<Kokkos::Schedule<Kokkos::Dynamic>, ExecSpace>;
using team_policy = Kokkos::TeamPolicy<Kokkos::Schedule<Kokkos::Dynamic>>;
using member_type = team_policy::member_type;
using ViewVectorType = Kokkos::View<int*, Kokkos::LayoutLeft, MemSpace>;

using cpu_exec_space = Kokkos::HostSpace::execution_space;
using cpu_range_policy = Kokkos::RangePolicy<Kokkos::Schedule<Kokkos::Dynamic>, cpu_exec_space>;

struct Graph {
    vector<vector<int>> adj;
    int vertices = 0;
    int edges = 0;

    int max_d = 0;  // maximum degree
    double avg_d = 0; // avg degree 
    int longest_shortest_path = 0; // longest shortest path 
};

Graph g;

// Performs Breadth-First Search from a given start vertex
// Returns the maximum distance reached from the start vertex
int BFS(int start) {
    int V = g.adj.size();
    vector<bool> visited(V, false);
    vector<int> distance(V, 0);
    
    queue<int> q;
    q.push(start);
    visited[start] = true;
    distance[start] = 0;
    
    int maxDistance = 0;
 
    while (!q.empty()) {
        int current = q.front();
        q.pop();
        
	if(g.adj[current].size() != 0){
		// printf("f");
            for (int neighbor : g.adj[current]) {
                if (!visited[neighbor]) {
                    visited[neighbor] = true;
                    distance[neighbor] = distance[current] + 1;
                    maxDistance = max(maxDistance, distance[neighbor]);
                    q.push(neighbor);
		}    
            }
        }
    }
    
    return maxDistance;
}

// Finds the longest shortest path in the graph using parallel BFS
// Uses OpenMP for parallelization
void find_longest_shortest_path() {
    int V = g.adj.size();
    int longest_shortest_path = 0;
    
    // vector<double> distances(V, 0);
    Kokkos::View<double*,Kokkos::HostSpace> distances("distances", V);

    // Perform BFS from each vertex and find the maximum distance
    // for (int i = 0; i < V; ++i) {
    Kokkos::parallel_for("BFS_loop", Kokkos::RangePolicy<Kokkos::OpenMP>(0,V), [=] (int i) {
        distances(i) = BFS(i);
    });

    for(int i =0; i < V; i++){
        if(distances(i) > longest_shortest_path)
            longest_shortest_path = distances(i);
    }

    g.longest_shortest_path = longest_shortest_path;
    return;
}

// Finds the maximum degree and the avg degree of the network
void find_degree(){
    int V = g.adj.size();

    Kokkos::View<int*,Kokkos::HostSpace> neighbors("neighbors", V);

    Kokkos::parallel_for("max_d_loop", Kokkos::RangePolicy<Kokkos::OpenMP>(0,V), [=] (int i) { 
        int size = g.adj[i].size();
        if(size != 0){
            neighbors(i) = g.adj[i].size();
        }	
    });
    
    int max_d = 0;
    int total = 0;
    for(int x = 0; x < V; x++){
        total += neighbors(x);
        if(max_d < neighbors(x)){
            max_d = neighbors(x);
        }
    }

    g.max_d = max_d;
    g.avg_d = (double) total / (double) V;
    
    return;
}

// Reads the graph from an input file specified in command line arguments
// Populates the global Graph structure 'g'
void read_graph(int argc, char* argv[]){
    if(argc != 3){
        cout << "Usage ./exec <Input File> <CSV file>" << endl;
        exit(1);
    }

    string inputString = argv[1];
    cout << "Dataset: " << inputString << endl;

    string line;
    ifstream inputFile(inputString);
    if(!inputFile){
        cout << "Failed to open the file" << endl;
        exit(1);
    }

    int maxNum;
    int firstInt, secondInt;
    while (getline(inputFile, line))
    {
        // check the commented lines    
        if(line[0] != '%'){
            istringstream iss(line);
            if(iss >> firstInt >> secondInt){
                //if(secondInt > firstInt){
                    maxNum = max(firstInt,secondInt);
                    if(maxNum >= g.vertices){
                        g.adj.resize(maxNum + 1);
                        g.vertices = maxNum + 1;
                    }
                    // check duplicate connection entries and check self loops
                    if (find(g.adj[firstInt].begin(), g.adj[firstInt].end(), secondInt) == g.adj[firstInt].end()
                       && firstInt != secondInt) {
                        // Edge doesn't exist yet, so add it
                        g.adj[firstInt].push_back(secondInt);
                        g.adj[secondInt].push_back(firstInt);
                        g.edges++;
                    }
                //}
            }else{
                printf("Failed to extract integers from line \n");
            }
        }
    }

}

string get_filename(const string& path) {
    // return filesystem::path(path).filename().string();

    // Find last occurrence of either forward slash or backslash
    size_t lastSlash = path.find_last_of("/\\");

    // If no slash found, return the entire path as it's just a filename
    if (lastSlash == string::npos) {
        return path;
    }

    // Return everything after the last slash
    return path.substr(lastSlash + 1);
}

// Creates a CSV file with appropriate headers if it doesn't exist
void create_CSV_if_not_exists(const string& csv_name) {
    ifstream fileCheck(csv_name);
    if (!fileCheck.good()) {
        ofstream file(csv_name);
        file << "Dataset,Vertices Count,Edges Count, Max_d, Avg_d, Longest Shortest Path,"
             << "Time_64,Time_128,Time_256,Time_512,Time_1024, Tricount\n";
        file.close();
    }
}

// Appends graph analysis results to the specified CSV file
void append_to_CSV(const string& csv_name, const string& dataset, const std::vector<double>& benchmark_results) {

    ofstream file(csv_name, ios::app);  // Open file in append mode
    
    if (!file.is_open()) {
        cerr << "Error: Unable to open file " << csv_name << endl;
        return;
    }
    
    string dataset_base; 
    dataset_base = get_filename(dataset);

    // Write data
    file << dataset_base << ","
         << g.vertices << ","
         << g.edges << ","
         << g.max_d << ","
         << g.avg_d << ","
         << g.longest_shortest_path;

    // Add benchmark results
    for (int x = 0; x < benchmark_results.size() - 1 ; x++){
        file << "," << benchmark_results[x];
    }
    // triangle count
    file << "," << (int) benchmark_results[5];

    file << "\n";
    
    file.close();
    
    std::cout << "Data for dataset '" << dataset << "' has been appended to " << csv_name << endl;
}

void create_CSR_format(ViewVectorType::HostMirror& h_indexPointerView, ViewVectorType::HostMirror& h_indicesView) {
    int V = g.adj.size();
    int count = 0;
    h_indexPointerView(0) = 0;
    for (int a = 0; a < V; a++) {
        // int size = graph.adj[a].size();
        // h_indexPointerView(a + 1) = h_indexPointerView(a) + size;
        // for (int b = 0; b < size; b++) {
        //     h_indicesView(count++) = graph.adj[a][b];
        // }
        int size = 0;
        for (int b = 0; b < g.adj[a].size(); b++){
            if(a < g.adj[a][b]){
                size++;
                h_indicesView(count++) = g.adj[a][b];
            }
            h_indexPointerView(a + 1) = h_indexPointerView(a) + size;
        }        
    }
}

//KOKKOS_FUNCTION
int count_triangles(ViewVectorType& indexPointerView, ViewVectorType& indicesView, int cudaNum) {
    int result = 0;
    Kokkos::parallel_reduce("Outer Reduction", team_policy(g.vertices - 1, cudaNum),
        KOKKOS_LAMBDA (const member_type& team, int& outer_reduction) {
            int inner_reduction_result = 0;
            int a = team.league_rank();
            int i = indexPointerView(a);
            int limit = indexPointerView(a + 1);

            Kokkos::parallel_reduce(Kokkos::TeamThreadRange(team, i, limit),
                [=] (int b, int& inner_reduction) {
                    int neighbor = indicesView(b);
                    int j = indexPointerView(neighbor);
                    int limit2 = indexPointerView(neighbor + 1);
                    int k = b;

                    while (k < limit && j < limit2) {
                        if (indicesView(k) > indicesView(j)) j++;
                        else if (indicesView(j) > indicesView(k)) k++;
                        else {
                            j++;
                            k++;
                            inner_reduction++;
                        }
                    }
                }, inner_reduction_result);

            Kokkos::single(Kokkos::PerTeam(team), [&] () {
                outer_reduction += inner_reduction_result;
            });
        }, result);
    return result;
}

std::vector<double> run_benchmark(ViewVectorType& indexPointerView, ViewVectorType& indicesView) {
    std::vector<int> cudaCounts = {64, 128, 256, 512, 1024};
    std::vector<double> results;
    int tri_count = 0;

    for (int cudaNum : cudaCounts) {
        std::vector<double> timings(10);
        for (int i = 0; i < 15; i++) {
            Kokkos::Timer timer;
            tri_count = count_triangles(indexPointerView, indicesView, cudaNum);
            double time = timer.seconds();
            if (i >= 5) {
                timings[i - 5] = time;
            }
            printf("Timing for execution %d: %f\n", i, time);
        }

        double avg = std::accumulate(timings.begin(), timings.end(), 0.0) / timings.size();
        results.push_back(avg);
        printf("\nAverage timing: %f\n", avg);
        printf("CUDA count: %d\n\n", cudaNum);
    }
    results.push_back(tri_count);
    return results;
}

int main(int argc, char* argv[]) {
    // Main function that orchestrates the graph analysis process
    // 1. Reads the graph from input file
    // 2. Calculates graph properties such as max_d, avg_d and longest shortest path
    // 3. Appends results to CSV file
    // 4. Prints results to console

    Kokkos::initialize();

    read_graph(argc, argv);
    find_degree();    
    find_longest_shortest_path();

    ViewVectorType indexPointerView("index Pointer view", g.vertices + 1);
    ViewVectorType indicesView("indices view", g.edges);

    ViewVectorType::HostMirror h_indexPointerView = Kokkos::create_mirror_view(indexPointerView);
    ViewVectorType::HostMirror h_indicesView = Kokkos::create_mirror_view(indicesView);

    create_CSR_format(h_indexPointerView, h_indicesView);

    Kokkos::deep_copy(indexPointerView, h_indexPointerView);
    Kokkos::deep_copy(indicesView, h_indicesView);

    vector<double> benchmark_results = run_benchmark(indexPointerView, indicesView);

    create_CSV_if_not_exists(argv[2]); // argv[2] countains the csv file path
    append_to_CSV(argv[2], argv[1], benchmark_results); // avg[1] contains the dataset

    cout << "Vertices Count: " << g.vertices << endl;
    cout << "Edges Count: " << g.edges << endl;
    cout << "max_d: " << g.max_d << endl;
    cout << "avg_d: " << g.avg_d << endl;
    cout << "Longest Shortest Path: " << g.longest_shortest_path << endl;
    cout << "\nTriangle Count: " << (int) benchmark_results[5] << endl;

    Kokkos::finalize();
    return 0;
}

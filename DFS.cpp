#include <iostream>
#include <vector>
#include <stack>
#include <omp.h>
using namespace std;

const int MAX = 100000;  // Maximum number of nodes
vector<int> graph[MAX];  // Adjacency list
bool visited[MAX];       // Visited array

void dfs(int start_node) {
    stack<int> s;
    s.push(start_node);

    while (!s.empty()) {
        int curr = s.top();
        s.pop();

        if (!visited[curr]) {
            visited[curr] = true;
            cout << curr << " ";

            // Traverse all neighbors in parallel
            #pragma omp parallel for
            for (int i = 0; i < graph[curr].size(); i++) {
                int neighbor = graph[curr][i];
                if (!visited[neighbor]) {
                    #pragma omp critical
                    {
                        s.push(neighbor);
                    }
                }
            }
        }
    }
}

int main() {
    int n, m, start;
    cout << "Enter number of nodes, edges, and start node: ";
    cin >> n >> m >> start;

    cout << "Enter edges (u v):" << endl;
    for (int i = 0; i < m; i++) {
        int u, v;
        cin >> u >> v;
        graph[u].push_back(v);
        graph[v].push_back(u);  // Undirected graph
    }

    // Initialize visited array
    #pragma omp parallel for
    for (int i = 0; i < n; i++) {
        visited[i] = false;
    }

    cout << "DFS Traversal: ";
    dfs(start);

    return 0;
}

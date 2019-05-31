#include <iostream>
#include <vector>

using namespace std;


bool saw(vector<vector<int> > check, int y, int x) 
{
    bool seen = 0;
    for (int i = 0; i < check.size(); ++i) {
        if (check[i][0] == y && check[i][1] == x) seen = 1;
    }
    return seen;
}

int main()
{
    int num;
    cin >> num;
    vector<vector<int> > seen;
    for (int i = 0; i < num; ++i) {
        cout << "RESTART\n";
        int dirn, row, col, posx, posy;
        cin >>dirn>>row>>col>>posy>>posx;
        vector <int> okay;
        okay.push_back(posy-1);
        okay.push_back(posx-1);
        seen.push_back(okay);
        for (int j = 0; j < dirn; ++j) {
            int x = posx;
            int y = posy;
            string ty;
            // cout << "ENTER CHAR";
            cin >> ty;
            // dir = getchar();
            // cout << "HERE" << dir;
            if (ty == "N") {
                while (!saw(seen, y, x)) ++y;
            } else if (ty == "S") {
                while (!saw(seen, y, x))
                    --y;
            } else if (ty == "E") {
                cout << "EAST";
                while (!saw(seen, y, x))
                    ++x;
            } else if (ty == "W") {
                while (!saw(seen, y, x))
                    --x;
            }
            okay.clear();
            okay.push_back(y);
            okay.push_back(x);
            seen.push_back(okay);
            posx = x;
            posy = y;
            cout << "COMP";
        }
        cout << posx << posy;
        seen.clear();
    }
}

/* Copyright (c) 2026 InstaDeep Ltd
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef _VEC_H_
#define _VEC_H_

#include <iostream>
#include <functional>
#include <iomanip>
#include <vector>
#include <cstdlib>
#include <list>
#include <cmath>
#include <algorithm>

namespace vec {

using std::vector;

template <typename T>
vector<T> linspace(int n, T begin=0, T end=1) {
    vector<T> out (n);
    T step = (end - begin) / n;
    T x = begin;
    for (int i = 0; i < n; i++) {
        out[i] = x;
        x += step;
    }
    return out;
}

/**
 *  Algebraic vector type, containing a std::vector.
 */
template <typename T>
class Vec {

    private:
        std::vector<T> _data;

    public:
        Vec (int n) :
            _data(std::vector<T>(n)) {}

        Vec (std::vector<T> v) :
            _data(v) {}

        Vec (std::initializer_list<T> list) {
            int n = list.size();
            _data = std::vector<T>(n);
            std::copy(list.begin(), list.end(), _data.begin());
        }

        T* data() {
            return _data.data();
        }

        // Iterators for agnostic loops and std::transform.
        using iterator = typename std::vector<T>::iterator;
        iterator begin() {return _data.begin();}
        iterator end() {return _data.end();}

        int size () {
            return _data.size();
        }

        T& operator [] (int i) {
            return _data[i];
        }

        Vec operator + (Vec b) {
            int n = size();
            Vec out = Vec(n);
            for (int i = 0; i < n; i++) {
                out[i] = _data[i] + b[i];
            }
            return out;
        }
        Vec operator + (T b) {
            int n = size();
            Vec<T> out (n);
            for (int i = 0; i < n; i++) {
                out[i] = _data[i] + b;
            }
            return out;
        }

        Vec operator * (Vec b) {
            int n = size();
            Vec out = Vec(n);
            for (int i = 0; i < n; i++) {
                out[i] = _data[i] * b[i];
            }
            return out;
        }
        Vec operator * (T b) {
            int n = size();
            Vec<T> out (n);
            for (int i = 0; i < n; i++) {
                out[i] = _data[i] * b;
            }
            return out;
        }

        Vec<bool> operator == (Vec b) {
            int n = size();
            Vec<bool> out (n);
            for (int i = 0; i < n; i++) {
                out[i] = (_data[i] == b[i]);
            }
            return out;
        }

        Vec repeat (int k) {
            int n = size();
            int nk = n * k;
            Vec out = Vec(nk);
            for (int i = 0; i < nk; i++) {
                out[i] = _data[i / k];
            }
            return out;
        }

        Vec tile (int k) {
            int n = size();
            int nk = n * k;
            Vec out = Vec(nk);
            for (int i = 0; i < nk; i++) {
                out[i] = _data[i % n];
            }
            return out;
        }

        template <typename B>
        Vec<B> map (std::function<B(T)> f) {
            Vec<B> out (size());
            std::transform(begin(), end(), out.begin(), f);
            return out;
        }

        template <typename B>
        Vec<B> cast() {
            int n = size();
            Vec<B> out(n);
            for (int i = 0; i < n; i++) {
                out[i] = static_cast<B>(_data[i]);
            }
            return out;
        }

        static Vec concat (std::list<Vec> vs) {
            int n = 0;
            for (Vec  &v: vs) {
                n += v.size();
            }
            Vec out (n);
            n = 0;
            for (Vec &v: vs) {
                std::copy(v.begin(), v.end(), out.begin() + n);
                n += v.size();
            }
            return out;
        }

        //------ static methods -----------

        static Vec zeros (int n) {
            Vec out (n);
            for (T& v: out) {v = 0;}
            return out;
        }

        static Vec ones (int n) {
            Vec out (n);
            for (T& v: out) {v = 1;}
            return out;
        }

        static bool allclose(Vec a, Vec b, T tol=0) {
            bool out = true;
            int n = a.size();
            for (int i = 0; i < n; i++) {
                out &= (std::abs(a[i] - b[i]) < tol);
            }
            return out;
        }

        void show(Vec a);
        void show(Vec a, int n_1);
        void show(Vec a, int n_2, int n_1);

};

template <typename T=int>
Vec<T> arange(int n) {
    Vec<T> out (n);
    for (int i = 0; i < n; i++) {
        out[i] = i;
    }
    return out;
}

template <typename T>
Vec<T> linspace(int n, T begin=0, T end=1) {
    Vec<T> out (n);
    T step = (end - begin) / n;
    T x = begin;
    for (int i = 0; i < n; i++) {
        out[i] = x;
        x += step;
    }
    return out;
}

// Transpose last 2 axes given penultimate (n_2) and last (n_1) axis sizes.
// This may be moved as a more generic `Array` method once we get rid of `Vec`.
template <typename T>
Vec<T> transpose_trailing_axes(Vec<T> vec, int n_2, int n_1) {
    int n_tot = vec.size();
    Vec<T> out (n_tot);
    int n_rows = n_tot / (n_2 * n_1);
    for (int r = 0; r < n_rows; r++) {
        for (int i = 0; i < n_2; i++) {
            for (int j = 0; j < n_1; j++) {
                int row_begin = r * n_1 * n_2;
                T val = vec[row_begin + i * n_1 + j];
                out[row_begin + j * n_2 + i] = val;
            }
        }
    }
    return out;
}

template <typename T>
std::ostream& showLine (std::ostream& out, T* v, int n, int width = 4) {
    for (int i = 0; i < n; i++) {
        out << std::setw(width) << v[i] << (i < n - 1 ? " " : "");
    }
    return out;
}

template <typename T>
std::ostream& showMatrix (std::ostream& out, T* v, int rows, int columns) {
    for (int r = 0; r < rows; r++) {
        out << (r == 0 ? "[" : " ");
        showLine(out, v + r * columns, columns);
        out << (r < rows - 1 ? "\n" : "]\n");
    }
    return out;
}

template <typename T>
std::ostream& show (std::ostream& out, T* v, int batches, int rows, int columns) {
    for (int b = 0; b < batches; b++) {
        out << (b == 0 ? "[\n " : " " );
        showMatrix(out, v, rows, columns);
    }
    return out << "]\n";

}

template <typename T>
std::ostream& show (std::ostream& out, Vec<T> vec) {
    return out << showLine(out, vec._data, vec.size());
}

template<typename T>
std::ostream& operator << (std::ostream& out, Vec<T>& v) {
    out << "Vec [";
    int n = v.size();
    for (int i = 0; i < n; i++) {
        out << v[i] << (i == n - 1 ? "]" : " ");
    }
    return out << "\n";
}

} // namespace vec

#endif //_VEC_H_

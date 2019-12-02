using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System.Linq;
using System;

public class gaussianNaiveBayes : MonoBehaviour
{
    // data
    double[][] x_train =
    {
        new double[] {2.4, 11.3},
        new double[] {3.2, 70.2},
        new double[] {75.7, 72.7},
        new double[] {2.8, 15.2}
    };
    string[] y_train = {"Y", "Y", "N", "N"};

    double[] test = {79.2, 12.1};

    string[] features = {"Distance", "Speed"};

    // model
    int N;
    Dictionary<string, int> Ny;
    Dictionary<string, Dictionary<string, double>> Mu, Sigma;


    void Start()
    {
        learn();
        showModel();
        predict();        
    }


    void learn() {
        Func<List<string>, Dictionary<string, int>> getCount = 
            (l) => l.GroupBy(
                x => x
                ).ToDictionary(
                    g => g.Key, 
                    g => g.Count()
                    );

        Func<string, int, List<double>> filter =
            (cl, at) => Enumerable.Range(0, y_train.Length).Where(
                i => y_train[i] == cl
                ).Select(
                    i => x_train[i][at]
                    ).ToList();

        N = x_train.Length;
        Ny = new Dictionary<string, int>();
        Ny = getCount(y_train.ToList());

        Mu = new Dictionary<string, Dictionary<string, double>>();
        Mu = Ny.Keys.ToDictionary(
            cl => cl,
            cl => Enumerable.Range(0, features.Length).ToDictionary(
                i => features[i],
                i => filter(cl, i).Average()
            )
        );

        Sigma = new Dictionary<string, Dictionary<string, double>>();
        Sigma = Ny.Keys.ToDictionary(
            cl => cl,
            cl => Enumerable.Range(0, features.Length).ToDictionary(
                i => features[i],
                i => filter(cl, i).Select(x => Math.Pow(x - Mu[cl][features[i]], 2)).Sum() / (Ny[cl] - 1)
            )
        );
    }

    void predict()
    {
        Func<string, string, double, double> gf =
            (cl, f, v) => Math.Exp(
                             -Math.Pow(v - Mu[cl][f], 2) / 2 / Sigma[cl][f]
                          ) / Math.Sqrt(2 * Math.PI * Sigma[cl][f]);

        Func<string, double> probability = 
            (cl) => (double)Ny[cl] / N * features.Zip(test, (f,v) => gf(cl, f, v)).Aggregate(
                    (x, y) => x * y);

        string pred = Ny.Keys.Select(
            cl => (probability(cl), cl)
            ).Max().Item2;

        Debug.Log("class(" + string.Join(", ", test) + "): " + pred);
    }

    void showModel() {
        string s = "Model:\n";
        s += "N: " + N.ToString() + "\n";

        s += "Ny:\n";
        foreach (var pair in Ny)
            s += "  " + pair.Key + ": " + pair.Value.ToString() + "\n";

        s += "Mu:\n";
        foreach (var cl in Mu) {
            s += "  " + cl.Key + "\n";
            foreach (var atr in cl.Value)
            {
                s += "    " + atr.Key + ": " + atr.Value.ToString() + "\n";
            }
        };


        s += "Sigma:\n";
        foreach (var cl in Sigma) {
            s += "  " + cl.Key + "\n";
            foreach (var atr in cl.Value)
            {
                s += "    " + atr.Key + ": " + atr.Value.ToString() + "\n";
            }
        };

        Debug.Log(s);
    }
}

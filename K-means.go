package K_means

import (
	"fmt"
	"math"
	"math/rand"
)

// Константы для обучения K-means
const (
	// KMeansK - количество кластеров
	KMeansK = 3
	// KMeansMaxIterations - максимальное количество итераций
	KMeansMaxIterations = 100
	// KMeansTolerance - допуск для сходимости
	KMeansTolerance = 1e-4
)

// KMeans представляет кластеризатор K-средних
type KMeans struct {
	K          int
	Centroids  [][]float64
	Labels     []int
	FeatureDim int
}

// NewKMeans создает новый кластеризатор K-средних
func NewKMeans(k int, featureDim int) *KMeans {
	return &KMeans{
		K:          k,
		Centroids:  make([][]float64, k),
		FeatureDim: featureDim,
	}
}

// InitializeCentroids инициализирует центроиды случайным образом
func (km *KMeans) InitializeCentroids(X [][]float64) {
	n := len(X)
	if n == 0 {
		return
	}

	// Находим диапазоны для каждой размерности
	mins := make([]float64, km.FeatureDim)
	maxs := make([]float64, km.FeatureDim)

	for i := 0; i < km.FeatureDim; i++ {
		mins[i] = math.Inf(1)
		maxs[i] = math.Inf(-1)
	}

	for _, x := range X {
		for i := 0; i < km.FeatureDim; i++ {
			if x[i] < mins[i] {
				mins[i] = x[i]
			}
			if x[i] > maxs[i] {
				maxs[i] = x[i]
			}
		}
	}

	// Инициализируем центроиды случайными значениями в пределах диапазонов
	for k := 0; k < km.K; k++ {
		km.Centroids[k] = make([]float64, km.FeatureDim)
		for i := 0; i < km.FeatureDim; i++ {
			km.Centroids[k][i] = mins[i] + rand.Float64()*(maxs[i]-mins[i])
		}
	}
}

// EuclideanDistance вычисляет евклидово расстояние между двумя векторами
func EuclideanDistance(a, b []float64) float64 {
	if len(a) != len(b) {
		return math.Inf(1)
	}

	sum := 0.0
	for i := range a {
		diff := a[i] - b[i]
		sum += diff * diff
	}
	return math.Sqrt(sum)
}

// FindClosestCentroid находит ближайший центроид для точки
func (km *KMeans) FindClosestCentroid(x []float64) int {
	minDist := math.Inf(1)
	closest := 0

	for k := 0; k < km.K; k++ {
		dist := EuclideanDistance(x, km.Centroids[k])
		if dist < minDist {
			minDist = dist
			closest = k
		}
	}

	return closest
}

// Fit обучает кластеризатор K-средних
func (km *KMeans) Fit(X [][]float64) {
	if len(X) == 0 {
		return
	}

	// Инициализация центроидов
	km.InitializeCentroids(X)
	km.Labels = make([]int, len(X))

	for iteration := 0; iteration < KMeansMaxIterations; iteration++ {
		// Шаг 1: Назначаем точки ближайшим центроидам
		changed := false
		for i, x := range X {
			closest := km.FindClosestCentroid(x)
			if km.Labels[i] != closest {
				changed = true
				km.Labels[i] = closest
			}
		}

		// Шаг 2: Обновляем центроиды
		newCentroids := make([][]float64, km.K)
		counts := make([]int, km.K)

		// Инициализируем новые центроиды
		for k := 0; k < km.K; k++ {
			newCentroids[k] = make([]float64, km.FeatureDim)
		}

		// Суммируем координаты точек в каждом кластере
		for i, x := range X {
			cluster := km.Labels[i]
			for j := 0; j < km.FeatureDim; j++ {
				newCentroids[cluster][j] += x[j]
			}
			counts[cluster]++
		}

		// Вычисляем средние значения
		maxCentroidShift := 0.0
		for k := 0; k < km.K; k++ {
			if counts[k] > 0 {
				for j := 0; j < km.FeatureDim; j++ {
					newCentroids[k][j] /= float64(counts[k])
				}
				// Вычисляем смещение центроида
				shift := EuclideanDistance(km.Centroids[k], newCentroids[k])
				if shift > maxCentroidShift {
					maxCentroidShift = shift
				}
				km.Centroids[k] = newCentroids[k]
			}
		}

		if !changed || maxCentroidShift < KMeansTolerance {
			fmt.Printf("K-means сошелся на итерации %d (максимальное смещение центроида: %.6f)\n", iteration, maxCentroidShift)
			break
		}

		if iteration%10 == 0 {
			fmt.Printf("Итерация %d: максимальное смещение центроида = %.6f\n", iteration, maxCentroidShift)
		}
	}
}

// Predict предсказывает кластер для точки
func (km *KMeans) Predict(x []float64) int {
	return km.FindClosestCentroid(x)
}

// Inertia вычисляет инерцию кластеризации (within-cluster sum of squares)
func (km *KMeans) Inertia(X [][]float64) float64 {
	inertia := 0.0
	for i, x := range X {
		cluster := km.Labels[i]
		dist := EuclideanDistance(x, km.Centroids[cluster])
		inertia += dist * dist
	}
	return inertia
}

// PrintClusterInfo выводит информацию о кластерах
func (km *KMeans) PrintClusterInfo(X [][]float64) {
	fmt.Println("\n=== Информация о кластерах ===")

	clusterCounts := make(map[int]int)
	for _, label := range km.Labels {
		clusterCounts[label]++
	}

	for k := 0; k < km.K; k++ {
		fmt.Printf("\nКластер %d:\n", k)
		fmt.Printf("  Количество точек: %d\n", clusterCounts[k])
		fmt.Printf("  Центроид: [")
		for i, val := range km.Centroids[k] {
			if i > 0 {
				fmt.Print(", ")
			}
			fmt.Printf("%.4f", val)
		}
		fmt.Println("]")
	}

	inertia := km.Inertia(X)
	fmt.Printf("\nИнерция (WCSS): %.4f\n", inertia)
}

package K_means

import (
	"fmt"
	"image/color"
	"math"
	"math/rand"

	"gonum.org/v1/plot"
	"gonum.org/v1/plot/plotter"
	"gonum.org/v1/plot/plotutil"
	"gonum.org/v1/plot/vg"
)

// KMeans представляет кластеризатор K-средних
type KMeans struct {
	K                   int
	Centroids           [][]float64
	Labels              []int
	FeatureDim          int
	KMeansMaxIterations int
	KMeansTolerance     float64
}

// NewKMeans создает новый кластеризатор K-средних
func NewKMeans(k int, featureDim int, KMeansMaxIterations int, KMeansTolerance float64) *KMeans {
	return &KMeans{
		K:                   k,
		Centroids:           make([][]float64, k),
		FeatureDim:          featureDim,
		KMeansMaxIterations: KMeansMaxIterations,
		KMeansTolerance:     KMeansTolerance,
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

	for iteration := 0; iteration < km.KMeansMaxIterations; iteration++ {
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

		if !changed || maxCentroidShift < km.KMeansTolerance {
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

// PlotClusters создает график кластеризации с отмеченными центроидами
// filename - имя файла для сохранения графика (например, "clusters.png")
// Для многомерных данных (>2 измерений) используются только первые 2 измерения
// Функцию можно вызывать как до обучения (покажет начальные центроиды),
// так и после обучения (покажет финальные центроиды и кластеры)
func (km *KMeans) PlotClusters(X [][]float64, filename string) error {
	if len(X) == 0 {
		return fmt.Errorf("нет данных для визуализации")
	}

	if km.FeatureDim < 2 {
		return fmt.Errorf("для визуализации необходимо минимум 2 измерения")
	}

	// Создаем новый график
	p := plot.New()
	// Определяем, обучена ли модель (есть ли Labels)
	if len(km.Labels) == len(X) {
		p.Title.Text = "K-means кластеризация (после обучения)"
	} else {
		p.Title.Text = "K-means кластеризация (до обучения - начальные центроиды)"
	}
	p.X.Label.Text = "X"
	p.Y.Label.Text = "Y"

	// Цвета для кластеров (RGBA с нормализованными значениями 0-1)
	plotColors := []color.Color{
		color.RGBA{R: 255, G: 0, B: 0, A: 255},     // Красный
		color.RGBA{R: 0, G: 0, B: 255, A: 255},     // Синий
		color.RGBA{R: 0, G: 255, B: 0, A: 255},     // Зеленый
		color.RGBA{R: 255, G: 165, B: 0, A: 255},   // Оранжевый
		color.RGBA{R: 128, G: 0, B: 128, A: 255},   // Фиолетовый
		color.RGBA{R: 139, G: 69, B: 19, A: 255},   // Коричневый
		color.RGBA{R: 255, G: 192, B: 203, A: 255}, // Розовый
		color.RGBA{R: 128, G: 128, B: 128, A: 255}, // Серый
	}

	// Определяем принадлежность точек к кластерам
	// Если Labels уже есть (после обучения), используем их
	// Иначе вычисляем на основе ближайших центроидов
	labels := make([]int, len(X))
	if len(km.Labels) == len(X) {
		// Используем существующие Labels
		copy(labels, km.Labels)
	} else {
		// Вычисляем принадлежность на основе ближайших центроидов
		for i, x := range X {
			labels[i] = km.FindClosestCentroid(x)
		}
	}

	// Рисуем точки для каждого кластера
	for k := 0; k < km.K; k++ {
		points := make(plotter.XYs, 0)
		for i, x := range X {
			if labels[i] == k {
				points = append(points, plotter.XY{
					X: x[0],
					Y: x[1],
				})
			}
		}

		if len(points) > 0 {
			scatter, err := plotter.NewScatter(points)
			if err != nil {
				return fmt.Errorf("ошибка создания scatter для кластера %d: %v", k, err)
			}

			// Используем цвет из палитры или черный по умолчанию
			if k < len(plotColors) {
				scatter.GlyphStyle.Color = plotColors[k]
			} else {
				scatter.GlyphStyle.Color = color.Black
			}
			scatter.GlyphStyle.Radius = vg.Points(3)
			scatter.Shape = plotutil.Shape(k % 4) // Разные формы для разных кластеров

			p.Add(scatter)
			p.Legend.Add(fmt.Sprintf("Кластер %d", k), scatter)
		}
	}

	// Рисуем центроиды специальными маркерами (звездочки)
	centroidPoints := make(plotter.XYs, km.K)
	for k := 0; k < km.K; k++ {
		centroidPoints[k] = plotter.XY{
			X: km.Centroids[k][0],
			Y: km.Centroids[k][1],
		}
	}

	centroidScatter, err := plotter.NewScatter(centroidPoints)
	if err != nil {
		return fmt.Errorf("ошибка создания scatter для центроидов: %v", err)
	}

	// Центроиды отображаем черными звездочками большего размера
	centroidScatter.GlyphStyle.Color = color.Black
	centroidScatter.GlyphStyle.Radius = vg.Points(8)
	centroidScatter.Shape = plotutil.Shape(4) // Звездочка

	p.Add(centroidScatter)
	p.Legend.Add("Центроиды", centroidScatter)

	// Сохраняем график в файл
	if err := p.Save(8*vg.Inch, 8*vg.Inch, filename); err != nil {
		return fmt.Errorf("ошибка сохранения графика: %v", err)
	}

	fmt.Printf("\nГрафик сохранен в файл: %s\n", filename)
	return nil
}

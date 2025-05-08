/*=========================================================================
 *
 *  Copyright UMC Utrecht and contributors
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *        http://www.apache.org/licenses/LICENSE-2.0.txt
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 *
 *=========================================================================*/

#ifndef itkImpactImageToImageMetric_h
#define itkImpactImageToImageMetric_h

#include "itkAdvancedImageToImageMetric.h"
#include "itkBSplineInterpolateImageFunction.h"
#include "itkBSplineInterpolateVectorImageFunction.h"
#include "itkImpactModelConfiguration.h"
#include "ImpactTensorUtils.h"
#include "ImpactLoss.h"

#include <torch/script.h>
#include <torch/torch.h>

#include <string>
#include <vector>
#include <random>

namespace itk
{

/** \class ImpactImageToImageMetric
 * \brief Semantic similarity metric for multimodal image registration based on deep features.
 *
 * This class is templated over the type of the fixed and moving images to be compared.
 *
 * Unlike conventional similarity metrics that rely on raw pixel intensities or handcrafted features,
 * this metric leverages high-level semantic representations extracted from pretrained deep learning models.
 * It enables robust registration by comparing the anatomical content of the fixed and moving images
 * in a shared feature space, rather than relying on potentially inconsistent intensity relationships.
 *
 * The semantic features are extracted from pretrained segmentation models (e.g., TotalSegmentator, SAM2.1)
 * and are used to guide the alignment of anatomical structures. These features are robust to noise,
 * artifacts, and intensity inhomogeneities, making the metric particularly effective in multimodal settings.
 *
 * The similarity is computed by comparing feature representations extracted from local image patches
 * (Jacobian mode) or from full feature maps (Static mode), using various distance functions (L1, L2, NCC, cosine).
 * In Jacobian mode, gradients are propagated through the feature extractor to enable efficient optimization.
 *
 * The proposed metric, called IMPACT (Image Metric with Pretrained model-Agnostic Comparison for Transmodality
 * registration), was shown to significantly improve alignment accuracy in several registration frameworks (Elastix,
 * VoxelMorph), across different anatomical regions and imaging modalities (CT, CBCT, MRI).
 *
 * Key characteristics:
 * - Semantic comparison based on deep features from pretrained segmentation networks.
 * - Robust to modality gaps, noise, and anatomical variability.
 * - Supports patch-based Jacobian mode and full-image Static mode.
 * - Compatible with various similarity distance functions.
 * - Fully integrated with multi-resolution strategies and weakly supervised mask-based optimization.
 *
 * \ingroup RegistrationMetrics
 * \ingroup Metrics
 */
template <typename TFixedImage, typename TMovingImage>
class ITK_TEMPLATE_EXPORT ImpactImageToImageMetric : public AdvancedImageToImageMetric<TFixedImage, TMovingImage>
{
public:
  ITK_DISALLOW_COPY_AND_MOVE(ImpactImageToImageMetric);

  /** Standard class typedefs. */
  using Self = ImpactImageToImageMetric;
  using Superclass = AdvancedImageToImageMetric<TFixedImage, TMovingImage>;
  using Pointer = SmartPointer<Self>;
  using ConstPointer = SmartPointer<const Self>;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Run-time type information (and related methods). */
  itkTypeMacro(ImpactImageToImageMetric, AdvancedImageToImageMetric);

  /** Typedefs from the superclass. */
  using typename Superclass::CoordinateRepresentationType;
  using typename Superclass::MovingImageType;
  using typename Superclass::MovingImagePixelType;
  using typename Superclass::MovingImageConstPointer;
  using typename Superclass::FixedImageType;
  using typename Superclass::FixedImageConstPointer;
  using typename Superclass::FixedImageRegionType;
  using typename Superclass::TransformType;
  using typename Superclass::TransformPointer;
  using typename Superclass::InputPointType;
  using typename Superclass::OutputPointType;
  using typename Superclass::TransformJacobianType;
  using typename Superclass::NumberOfParametersType;
  using typename Superclass::InterpolatorType;
  using typename Superclass::InterpolatorPointer;
  using typename Superclass::RealType;
  using typename Superclass::GradientPixelType;
  using typename Superclass::GradientImageType;
  using typename Superclass::GradientImagePointer;
  using typename Superclass::FixedImageMaskType;
  using typename Superclass::FixedImageMaskPointer;
  using typename Superclass::MovingImageMaskType;
  using typename Superclass::MovingImageMaskPointer;
  using typename Superclass::MeasureType;
  using typename Superclass::DerivativeType;
  using typename Superclass::DerivativeValueType;
  using typename Superclass::ParametersType;
  using typename Superclass::FixedImagePixelType;
  using typename Superclass::MovingImageRegionType;
  using typename Superclass::ImageSamplerType;
  using typename Superclass::ImageSamplerPointer;
  using typename Superclass::ImageSampleContainerType;
  using typename Superclass::ImageSampleContainerPointer;
  using typename Superclass::FixedImageLimiterType;
  using typename Superclass::MovingImageLimiterType;
  using typename Superclass::FixedImageLimiterOutputType;
  using typename Superclass::MovingImageLimiterOutputType;
  using typename Superclass::MovingImageDerivativeScalesType;
  using typename Superclass::ThreadInfoType;

  /** The fixed image dimension. */
  itkStaticConstMacro(FixedImageDimension, unsigned int, FixedImageType::ImageDimension);

  /** The moving image dimension. */
  itkStaticConstMacro(MovingImageDimension, unsigned int, MovingImageType::ImageDimension);

  /** Compute the similarity value (loss) for a given transformation parameter set.
   * This method is intended for use with single-valued optimizers in a single-threaded context.
   * It is typically used in testing or debugging scenarios.
   */
  virtual MeasureType
  GetValueSingleThreaded(const ParametersType & parameters) const;

  /** Compute the similarity value (loss) for a given transformation parameter set.
   * This is the main entry point for single-valued optimizers and is multi-threaded internally.
   * It aggregates the contribution from all threads.
   */
  MeasureType
  GetValue(const ParametersType & parameters) const override;

  /** Compute the gradient (derivative) of the similarity value with respect to transformation parameters.
   * Used in gradient-based optimization methods. Internally supports multi-threaded computation.
   */
  void
  GetDerivative(const ParametersType & parameters, DerivativeType & derivative) const override;

  /** Compute both the similarity value and its gradient in a single-threaded context.
   */
  void
  GetValueAndDerivativeSingleThreaded(const ParametersType & parameters,
                                      MeasureType &          value,
                                      DerivativeType &       derivative) const;

  /** Compute both the similarity value and its gradient in a multi-threaded context.
   * This is the main function called by optimizers requiring both value and derivative,
   * and it supports full parallel execution.
   */
  void
  GetValueAndDerivative(const ParametersType & parameters,
                        MeasureType &          value,
                        DerivativeType &       derivative) const override;

  /**
   * Initializes the metric and loads models, interpolators, and feature map settings.
   * Called before the optimization loop starts. Ensures all configuration dependencies are resolved.
   */
  void
  Initialize() override;

  /** Set/Get the list of TorchScript model configurations used to extract features from the fixed image.
   * Each model can target a different resolution, architecture, or semantic level.
   */
  itkSetMacro(FixedModelsConfiguration, std::vector<ImpactModelConfiguration>);
  itkGetConstMacro(FixedModelsConfiguration, std::vector<ImpactModelConfiguration>);

  /** Set/Get the list of TorchScript model configurations used to extract features from the moving image.
   * Allows using different models for fixed and moving images to support asymmetric or multimodal setups.
   */
  itkSetMacro(MovingModelsConfiguration, std::vector<ImpactModelConfiguration>);
  itkGetConstMacro(MovingModelsConfiguration, std::vector<ImpactModelConfiguration>);

  /** Set/Get the subset of feature indices to be used in the loss computation.
   * This allows dimensionality reduction or focusing on the most informative channels.
   */
  itkSetMacro(SubsetFeatures, std::vector<unsigned int>);
  itkGetConstMacro(SubsetFeatures, std::vector<unsigned int>);

  /** Set/Get the weights applied to each layer's loss contribution.
   * Useful for balancing the influence of layers with different semantic granularity.
   */
  itkSetMacro(LayersWeight, std::vector<float>);
  itkGetConstMacro(LayersWeight, std::vector<float>);


  /** Set/Get the type of loss function used for each layer (e.g., "l1", "cosine", "ncc").
   * Supports heterogeneous losses across layers to adapt to the nature of each feature representation.
   */
  itkSetMacro(Distance, std::vector<std::string>);
  itkGetConstMacro(Distance, std::vector<std::string>);

  /** Set/Get the number of principal components to keep after applying PCA to the feature maps.
   * Set to 0 to disable PCA. Reduces dimensionality and improve runtime.
   */
  itkSetMacro(PCA, std::vector<unsigned int>);
  itkGetConstMacro(PCA, std::vector<unsigned int>);

  /** Set/Get the device on which all model inference and tensor operations are performed.
   * Example: torch::Device(torch::kCUDA, 0) for GPU 0.
   */
  itkSetMacro(Device, torch::Device);
  itkGetConstMacro(Device, torch::Device);

  /** Set/Get whether the extracted feature maps should be written to disk (for inspection or debugging).
   * Useful for visualizing the intermediate representations used by the metric.
   */
  itkSetMacro(WriteFeatureMaps, bool);
  itkGetConstMacro(WriteFeatureMaps, bool);

  /** Set/Get the directory path where feature maps will be written if WriteFeatureMaps is true.
   * The path will be created if it does not exist.
   */
  itkSetMacro(FeatureMapsPath, std::string);
  itkGetConstMacro(FeatureMapsPath, std::string);

  /** Set/Get the mode of operation: "Jacobian", "Static", or "Dynamic".
   * - "Jacobian": online patch extraction with gradient backpropagation.
   * - "Static": precomputed full feature maps.
   */
  itkSetMacro(Mode, std::string);
  itkGetConstMacro(Mode, std::string);

  /** Set/Get the current resolution level
   */
  itkSetMacro(CurrentLevel, unsigned int);
  itkGetConstMacro(CurrentLevel, unsigned int);

  /** Set/Get the manual seed
   */
  itkSetMacro(Seed, unsigned int);
  itkGetConstMacro(Seed, unsigned int);

  /** Set/Get how often (in number of optimizer iterations) the feature maps should be updated.
   * A value of 0 disables updates (useful in static mode). Positive values enable periodic refreshes.
   */
  itkSetMacro(FeaturesMapUpdateInterval, int);
  itkGetConstMacro(FeaturesMapUpdateInterval, int);

protected:
  ImpactImageToImageMetric();
  ~ImpactImageToImageMetric() override = default;

  /**
   * Initializes per-thread loss structures and ensures thread safety for parallel execution.
   * Overrides superclass method because the metric uses its own loss aggregation system.
   */
  void
  InitializeThreadingParameters() const override;

  /** Protected Typedefs ******************/

  /**
   * Thread-local structure that accumulates loss values and gradients for each layer.
   *
   * Encapsulates one loss object per output layer (as defined by layersMask), allowing multi-layer
   * loss computation and weighted aggregation. Also provides interfaces to get final loss and gradient.
   */
  struct LossPerThreadStruct
  {
    std::vector<std::unique_ptr<ImpactLoss::Loss>> m_losses;
    std::vector<float>                             m_layersWeight;
    SizeValueType                                  m_numberOfPixelsCounted;
    int                                            m_nb_parameters;
    std::mt19937                                   m_randomGenerator;

    void
    init(std::vector<std::string> distance_name, std::vector<float> layersWeight, unsigned int seed)
    {
      if (seed > 0)
      {
        this->m_randomGenerator = std::mt19937(seed);
      }
      else
      {
        this->m_randomGenerator = std::mt19937(time(nullptr));
      }
      this->m_layersWeight = layersWeight;
      for (std::string name : distance_name)
      {
        m_losses.push_back(ImpactLoss::LossFactory::Instance().Create(name));
      }
    }

    void
    set_nb_parameters(int nb_parameters)
    {
      this->m_nb_parameters = nb_parameters;
      for (int l = 0; l < this->m_layersWeight.size(); ++l)
      {
        this->m_losses[l]->set_nb_parameters(nb_parameters);
      }
    }

    void
    reset()
    {
      this->m_numberOfPixelsCounted = 0;
      for (std::unique_ptr<ImpactLoss::Loss> & loss : m_losses)
      {
        loss->reset();
      }
    }

    double
    GetValue()
    {
      MeasureType value = MeasureType{};
      for (int l = 0; l < this->m_layersWeight.size(); ++l)
      {
        value +=
          this->m_layersWeight[l] * this->m_losses[l]->GetValue(static_cast<double>(this->m_numberOfPixelsCounted));
      }
      return value;
    }

    DerivativeType
    GetDerivative()
    {
      DerivativeType derivative = DerivativeType(this->m_nb_parameters);
      derivative.Fill(DerivativeValueType{});
      for (int l = 0; l < this->m_layersWeight.size(); ++l)
      {
        torch::Tensor d = this->m_layersWeight[l] *
                          this->m_losses[l]->GetDerivative(static_cast<double>(this->m_numberOfPixelsCounted));
        for (int i = 0; i < d.size(0); ++i)
        {
          derivative[i] += d[i].item<float>();
        }
      }
      return derivative;
    }

    LossPerThreadStruct &
    operator+=(const LossPerThreadStruct & other)
    {
      const auto * lossPerThreadStructOther = dynamic_cast<const LossPerThreadStruct *>(&other);
      if (lossPerThreadStructOther)
      {
        m_numberOfPixelsCounted += lossPerThreadStructOther->m_numberOfPixelsCounted;
        for (int i = 0; i < lossPerThreadStructOther->m_losses.size(); ++i)
        {
          *m_losses[i] += *lossPerThreadStructOther->m_losses[i];
        }
      }
      return *this;
    }
  };

  /** Typedefs inherited from superclass */
  using typename Superclass::FixedImageIndexType;
  using typename Superclass::FixedImageIndexValueType;
  using typename Superclass::MovingImageIndexType;
  using typename Superclass::FixedImagePointType;
  using typename Superclass::MovingImagePointType;
  using typename Superclass::MovingImageContinuousIndexType;
  using typename Superclass::BSplineInterpolatorType;
  using typename Superclass::MovingImageDerivativeType;
  using typename Superclass::NonZeroJacobianIndicesType;

  /** Check if a patch centered at the given fixed image point is valid for sampling.
   * This version considers a specific patch layout (patchIndex) and verifies that all
   * transformed points remain inside the moving image domain.
   */
  bool
  SampleCheck(const FixedImagePointType &             fixedImageCenterCoordinate,
              const std::vector<std::vector<float>> & patchIndex) const;

  /** Check if the fixed image point lies within valid bounds for sampling.
   * This version does not consider patch geometry. It is used to validate
   * isolated points before processing them in the similarity metric.
   */
  bool
  SampleCheck(const FixedImagePointType & fixedImageCenterCoordinate) const;

  /** Compute the similarity value contribution for a given thread.
   * This method is called in parallel across threads. Each thread accumulates
   * a partial loss.
   */
  void
  ThreadedGetValue(ThreadIdType threadID) const override;

  /** Combine the similarity values computed by all threads.
   * Aggregates the loss contributions stored in each thread’s `LossPerThreadStruct`
   * into a global scalar value used by the optimizer.
   */
  void
  AfterThreadedGetValue(MeasureType & value) const override;

  /** Compute both similarity value and its derivative (gradient) for a given thread.
   * Each thread computes the semantic loss and its gradient w.r.t. transformation parameters.
   * Gradients are computed either analytically (Jacobian mode) or skipped (static mode).
   */
  void
  ThreadedGetValueAndDerivative(ThreadIdType threadID) const override;

  /** Combine the values and gradients computed by all threads.
   * Final reduction step to produce global loss and gradient vectors used in optimization.
   */
  void
  AfterThreadedGetValueAndDerivative(MeasureType & value, DerivativeType & derivative) const override;

  /** Compute the semantic similarity value using the current transform parameters.
   * This method evaluates the loss at all sampled points using the current transformation,
   * without computing derivatives. It uses patch-based inference and feature comparison.
   * Applicable in both Jacobian and static modes.
   *
   * \param fixedPoints Sampled points in the fixed image.
   * \param loss Loss objects (one per semantic layer) to accumulate values.
   * \return Number of valid samples used in the computation.
   */
  unsigned int
  ComputeValue(const std::vector<FixedImagePointType> & fixedPoints, LossPerThreadStruct & loss) const;

  /** Compute the semantic similarity value in static mode (precomputed feature maps).
   * Unlike ComputeValue(), this version uses pre-extracted static features for both
   * fixed and moving images, avoiding repeated forward passes through the model.
   *
   * \param fixedPoints Sampled points in the fixed image.
   * \param loss Loss objects (one per semantic layer) to accumulate values.
   * \return Number of valid samples used in the computation.
   */
  unsigned int
  ComputeValueStatic(const std::vector<FixedImagePointType> & fixedPoints, LossPerThreadStruct & loss) const;

  /** Compute both the semantic similarity value and its derivative using Jacobian mode.
   * In this mode, gradients are backpropagated through the model to compute the
   * sensitivity of the metric to transformation parameters. This is essential for
   * enabling gradient-based optimization.
   *
   * \param fixedPoints Sampled points in the fixed image.
   * \param loss Loss objects to store both values and gradients per layer.
   * \return Number of valid samples used in the computation.
   */
  unsigned int
  ComputeValueAndDerivativeJacobian(const std::vector<FixedImagePointType> & fixedPoints,
                                    LossPerThreadStruct &                    loss) const;

  /** Compute value and derivative in static mode (precomputed features).
   * Gradients are computed via chain rule using the interpolated feature fields.
   *
   * \param fixedPoints Sampled points in the fixed image.
   * \param loss Loss objects to store both values and gradients per layer.
   * \return Number of valid samples used in the computation.
   */
  unsigned int
  ComputeValueAndDerivativeStatic(const std::vector<FixedImagePointType> & fixedPoints,
                                  LossPerThreadStruct &                    loss) const;


  /** Update the fixed feature maps (static mode).
   * Re-extracts deep features from the images using the TorchScript models
   * when entering a new pyramid level or after a feature update interval.
   */
  void
  UpdateFeaturesMaps();

  /** Update the moving feature maps (static mode).
   * Same as UpdateFeaturesMaps(), but applied to the moving image.
   */
  void
  UpdateMovingFeaturesMaps();

private:
  /** Interpolator for fixed image intensities, using B-spline of order 3 (double precision). */
  using FixedInterpolatorType = BSplineInterpolateImageFunction<FixedImageType, CoordinateRepresentationType, double>;
  /** Feature maps are stored as VectorImages of floats with same dimension as fixed image. */
  using FeaturesImageType = itk::VectorImage<float, FixedImageDimension>;
  /** Interpolator for feature maps (vector-valued), using scalar B-spline interpolation. */
  using FeaturesInterpolatorType = BSplineInterpolateVectorImageFunction<
    FeaturesImageType,
    BSplineInterpolateImageFunction<itk::Image<float, FixedImageDimension>, CoordinateRepresentationType, float>>;

  /**
   * \struct FeaturesMaps
   * Encapsulates both the feature map image and its associated interpolator.
   * This allows evaluating feature vectors (and derivatives) at arbitrary points.
   */
  struct FeaturesMaps
  {
    typename FeaturesImageType::Pointer m_featuresMaps;
    FeaturesInterpolatorType            m_featuresMapsInterpolator;

    FeaturesMaps(typename FeaturesImageType::Pointer featuresMaps)
      : m_featuresMaps(featuresMaps)
    {
      this->m_featuresMapsInterpolator = FeaturesInterpolatorType();
      this->m_featuresMapsInterpolator.SetInputImage(featuresMaps);
    }
  };

  using FeaturesMaps = typename ImpactImageToImageMetric<TFixedImage, TMovingImage>::FeaturesMaps;

  /**
   * Extracts a fixed image patch tensor centered at a point, using the precomputed patchIndex.
   * Interpolation is performed using the fixed image interpolator.
   */
  torch::Tensor
  EvaluateFixedImagesPatchValue(const FixedImagePointType &             fixedImageCenterCoordinate,
                                const std::vector<std::vector<float>> & patchIndex,
                                const std::vector<int64_t> &            patchSize) const;

  /**
   * Extracts a moving image patch tensor (intensity values) corresponding to a fixed point,
   * using the transform and moving image interpolator.
   */
  torch::Tensor
  EvaluateMovingImagesPatchValue(const FixedImagePointType &             fixedImageCenterCoordinate,
                                 const std::vector<std::vector<float>> & patchIndex,
                                 const std::vector<int64_t> &            patchSize) const;

  /**
   * Extracts moving image patch values *and* computes the spatial Jacobians w.r.t. image coordinates.
   * Used in Jacobian mode for backpropagating through the transform.
   */
  torch::Tensor
  EvaluateMovingImagesPatchValuesAndJacobians(const FixedImagePointType &             fixedImageCenterCoordinate,
                                              torch::Tensor &                         movingImagesPatchesJacobians,
                                              const std::vector<std::vector<float>> & patchIndex,
                                              const std::vector<int64_t> &            patchSize,
                                              int                                     s) const;

  /**
   * Given a list of fixed points and model configurations, generates valid patch indices
   * and filters out invalid points (outside mask/boundary). Returns filtered fixed points.
   */
  template <typename ImagePointType>
  std::vector<ImagePointType>
  GeneratePatchIndex(const std::vector<ImpactModelConfiguration> &               modelConfig,
                     std::mt19937 &                                              randomGenerator,
                     const std::vector<ImagePointType> &                         fixedPointsTmp,
                     std::vector<std::vector<std::vector<std::vector<float>>>> & patchIndex) const;

  /** TorchScript model configurations for fixed and moving image feature extraction. */
  std::vector<ImpactModelConfiguration> m_FixedModelsConfiguration;
  std::vector<ImpactModelConfiguration> m_MovingModelsConfiguration;

  std::vector<unsigned int> m_SubsetFeatures;
  std::vector<unsigned int> m_PCA;
  std::vector<float>        m_LayersWeight;
  std::vector<std::string>  m_Distance;
  int                       m_FeaturesMapUpdateInterval;
  std::string               m_Mode;
  bool                      m_WriteFeatureMaps;
  std::string               m_FeatureMapsPath;
  torch::Device             m_Device = torch::Device(torch::kCPU);
  unsigned int              m_CurrentLevel;
  unsigned int              m_Seed;


  std::vector<FeaturesMaps>  m_fixedFeaturesMaps;
  std::vector<FeaturesMaps>  m_movingFeaturesMaps;
  std::vector<torch::Tensor> m_principal_components;

  std::vector<std::vector<unsigned int>> m_features_indexes;


  /**
   * Interpolator for fixed image intensity values, set once at initialization.
   * Uses 3rd-order B-spline interpolation.
   */
  InterpolatorPointer m_fixedInterpolator = [this] {
    const auto interpolator = FixedInterpolatorType::New();
    interpolator->SetSplineOrder(3);
    return interpolator;
  }();

  std::vector<unsigned int>
  GetSubsetOfFeatures(const std::vector<unsigned int> & features_index, std::mt19937 & randomGenerator, int n) const;

  /** Thread-safe wrapper for per-thread loss computation (padded to avoid false sharing). */
  itkPadStruct(ITK_CACHE_LINE_ALIGNMENT, LossPerThreadStruct, PaddedLossPerThreadStruct);

  itkAlignedTypedef(ITK_CACHE_LINE_ALIGNMENT, PaddedLossPerThreadStruct, AlignedLossPerThreadStruct);

  /** Per-thread loss structures, dynamically allocated during initialization. */
  mutable std::unique_ptr<AlignedLossPerThreadStruct[]> m_LossThreadStruct{ nullptr };

  mutable int m_LossThreadStructSize = 0;
};

} // end namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#  include "itkImpactImageToImageMetric.hxx"
#endif

#endif // end #ifndef itkImpactImageToImageMetric_h

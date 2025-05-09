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

/**
 * \class ImpactImageToImageMetric
 * \brief A semantic similarity metric for multimodal image registration based on deep learning features.
 *
 * This class define a loss by compares the fixed and moving images using high-level semantic representations
 * extracted from pretrained deep learning models, rather than relying on raw pixel intensities
 * or handcrafted features.
 *
 * All details can be found in:\n
 * Valentin Boussot, Cédric Hémon, Jean-Claude Nunes, Jason Dowling, Simon Rouzé, Caroline Lafond, Anaïs Barateau,
 * Jean-Louis Dillenseger A Generic Semantic Loss for Multimodal Image Registration https://arxiv.org/abs/2503.24121
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
  itkGetConstReferenceMacro(FixedModelsConfiguration, std::vector<ImpactModelConfiguration>);

  /** Set/Get the list of TorchScript model configurations used to extract features from the moving image.
   * Allows using different models for fixed and moving images to support asymmetric or multimodal setups.
   */
  itkSetMacro(MovingModelsConfiguration, std::vector<ImpactModelConfiguration>);
  itkGetConstReferenceMacro(MovingModelsConfiguration, std::vector<ImpactModelConfiguration>);

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

  /**
   * Set/Get whether mixed precision (float16/float32) should be used during model inference.
   */
  itkSetMacro(UseMixedPrecision, bool);
  itkGetConstMacro(UseMixedPrecision, bool);

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
   * \brief Thread-local structure for accumulating loss values and gradients for each layer.
   *
   * This structure encapsulates one loss object per output layer (defined by the layersMask), enabling
   * multi-layer loss computation and weighted aggregation. It allows efficient computation of the final loss
   * and its gradients by keeping track of the contributions from each layer.
   *
   * \details This structure is designed to be used in a multi-threaded environment, where each thread
   * maintains its own instance to store intermediate results, ensuring thread-safety during parallel loss
   * and gradient computations.
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

  /**
   * \brief Checks if a patch centered at the given fixed image point is valid for sampling.
   *
   * This function verifies that a patch, defined by the `patchIndex`, centered at the given fixed image
   * point, is valid for sampling. It ensures that all transformed points of the patch remain inside
   * the domain of the moving image, or within the moving mask if one is defined.
   * It is used to validate patches before they are processed in the similarity metric.
   *
   * \param fixedImageCenterCoordinate The center coordinate of the patch in the fixed image.
   * \param patchIndex A layout of the patch, defining the indices of the region to sample around
   *                   the fixed image point.
   *
   * \return `true` if the patch is valid for sampling, meaning all transformed points are inside the
   *         moving image domain or mask; `false` otherwise.
   */
  bool
  SampleCheck(const FixedImagePointType &             fixedImageCenterCoordinate,
              const std::vector<std::vector<float>> & patchIndex) const;

  /**
   * \brief Checks if the fixed image point lies within valid bounds for sampling.
   *
   * This function checks if a fixed image point is within the valid bounds for sampling.
   * It is used to validate individual points before they are processed in the similarity metric.
   *
   * \param fixedImageCenterCoordinate The coordinate of the fixed image point to validate.
   *
   * \return `true` if the fixed image point is within valid bounds for sampling, `false` otherwise.
   */
  bool
  SampleCheck(const FixedImagePointType & fixedImageCenterCoordinate) const;

  /**
   * \brief Computes the similarity value contribution for a given thread.
   *
   * This method is called in parallel across multiple threads. Each thread calculates
   * and accumulates its partial loss contribution to the overall similarity value.
   *
   * \param threadID The unique identifier of the thread processing the similarity calculation.
   */
  void
  ThreadedGetValue(ThreadIdType threadID) const override;

  /**
   * \brief Combines the similarity values computed by all threads.
   *
   * This method aggregates the loss contributions stored in each thread’s `LossPerThreadStruct`
   * and combines them into a global scalar value. This aggregated value is then used by the optimizer
   * during the optimization process.
   *
   * \param value The scalar value to store the aggregated similarity result.
   */
  void
  AfterThreadedGetValue(MeasureType & value) const override;

  /**
   * \brief Computes both the similarity value and its gradient for a given thread.
   *
   * Each thread computes the semantic loss and its gradient with respect to the transformation parameters.
   *
   * \param threadID The unique identifier of the thread performing the calculation.
   */
  void
  ThreadedGetValueAndDerivative(ThreadIdType threadID) const override;

  /**
   * \brief Combines the values and gradients computed by all threads.
   *
   * This method performs the final reduction step to aggregate the values and gradients
   * computed by each thread into global loss and gradient vectors. These aggregated results
   * are then used in the optimization process.
   *
   * \param value The global loss value computed by combining the contributions from all threads.
   * \param derivative The global gradient vector computed by combining the gradients from all threads.
   */
  void
  AfterThreadedGetValueAndDerivative(MeasureType & value, DerivativeType & derivative) const override;

  /**
   * \brief Computes the semantic similarity value using the current transform parameters.
   *
   * This method computes the loss at all sampled points using the current transformation parameters,
   * without calculating derivatives. It performs patch-based inference and feature comparison in
   * Jacobian mode.
   *
   * \param fixedPoints A vector of sampled points in the fixed image.
   * \param loss The loss objects (one per layer) where the values will be accumulated.
   *
   * \return The number of valid samples used in the similarity computation.
   */
  unsigned int
  ComputeValue(const std::vector<FixedImagePointType> & fixedPoints, LossPerThreadStruct & loss) const;

  /**
   * \brief Computes the semantic similarity value in static mode (using precomputed feature maps).
   *
   * Unlike `ComputeValue()`, this method uses pre-extracted static feature maps for both the
   * fixed and moving images, avoiding repeated forward passes through the model
   *
   * \param fixedPoints A vector of sampled points in the fixed image.
   * \param loss The loss objects (one per semantic layer) where the values will be accumulated.
   *
   * \return The number of valid samples used in the similarity computation.
   */
  unsigned int
  ComputeValueStatic(const std::vector<FixedImagePointType> & fixedPoints, LossPerThreadStruct & loss) const;

  /**
   * \brief Computes both the semantic similarity value and its derivative using Jacobian mode.
   *
   * This method computes both the similarity value and its derivative (gradient)
   * by backpropagating through the model. This allows the metric to assess the sensitivity to
   * transformation parameters, which is crucial for gradient-based optimization.
   *
   * \param fixedPoints A vector of sampled points in the fixed image.
   * \param loss Loss objects to store both the values and gradients for each semantic layer.
   *
   * \return The number of valid samples used in the similarity and gradient computation.
   */
  unsigned int
  ComputeValueAndDerivativeJacobian(const std::vector<FixedImagePointType> & fixedPoints,
                                    LossPerThreadStruct &                    loss) const;

  /**
   * \brief Computes the value and derivative in static mode (using precomputed features).
   *
   * This method computes the similarity value and its derivative (gradient)
   * using precomputed feature maps. Gradients are computed via the chain rule, using the
   * interpolated feature fields rather than backpropagating through the model.
   *
   * \param fixedPoints A vector of sampled points in the fixed image.
   * \param loss Loss objects to store both the values and gradients for each semantic layer.
   *
   * \return The number of valid samples used in the similarity and gradient computation.
   */
  unsigned int
  ComputeValueAndDerivativeStatic(const std::vector<FixedImagePointType> & fixedPoints,
                                  LossPerThreadStruct &                    loss) const;


  /**
   * \brief Updates the fixed feature maps in static mode.
   *
   * This method re-extracts deep features from the images using the TorchScript models
   * when transitioning to a new pyramid level or after a specified feature update interval.
   * This ensures that the feature maps are kept up-to-date for the registration process.
   */
  void
  UpdateFeaturesMaps();

  /**
   * \brief Updates the moving feature maps in static mode.
   *
   * This method performs the same operation as `UpdateFeaturesMaps()`, but it applies to the moving image.
   * It re-extracts deep features from the moving image using the TorchScript models, ensuring that the
   * feature maps are updated when transitioning to a new pyramid level or after a feature update interval.
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
   * \brief Encapsulates a feature map image and its associated interpolator.
   *
   * This structure holds both the feature map image and the corresponding interpolator, enabling
   * the evaluation of feature vectors (and their derivatives) at arbitrary points in the image.
   * It is designed to facilitate feature extraction and interpolation during image registration in static mode.
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
   * \brief Extracts a fixed image patch tensor centered at a given point using the precomputed patch index.
   *
   * This method extracts a patch from the fixed image centered at the specified point,
   * based on the provided `patchIndex` and `patchSize`. Interpolation is performed using
   * the fixed image interpolator to sample the feature values at the specified coordinates.
   *
   * \param fixedImageCenterCoordinate The coordinate of the center point of the patch in the fixed image.
   * \param patchIndex A vector defining the relative indices for extracting the patch.
   * \param patchSize The size of the patch to extract.
   *
   * \return A tensor representing the extracted patch from the fixed image.
   */
  torch::Tensor
  EvaluateFixedImagesPatchValue(const FixedImagePointType &             fixedImageCenterCoordinate,
                                const std::vector<std::vector<float>> & patchIndex,
                                const std::vector<int64_t> &            patchSize) const;

  /**
   * \brief Extracts a moving image patch tensor (intensity values) corresponding to a fixed point.
   *
   * This method extracts a patch from the moving image, centered at the corresponding transformed
   * point from the fixed image. The extraction is performed using the provided transform and
   * the moving image interpolator to sample intensity values at the transformed coordinates.
   *
   * \param fixedImageCenterCoordinate The coordinate of the center point in the fixed image.
   * \param patchIndex A 2D vector defining the relative indices for extracting the patch.
   * \param patchSize The size of the patch to extract.
   *
   * \return A tensor representing the extracted patch from the moving image.
   */
  torch::Tensor
  EvaluateMovingImagesPatchValue(const FixedImagePointType &             fixedImageCenterCoordinate,
                                 const std::vector<std::vector<float>> & patchIndex,
                                 const std::vector<int64_t> &            patchSize) const;

  /**
   * \brief Extracts moving image patch values and computes the spatial Jacobians with respect to image coordinates.
   *
   * This method extracts a patch from the moving image centered at the corresponding transformed
   * point from the fixed image. It also computes the image gradient. This is used in Jacobian mode to enable
   * gradient computation via the chain rule during optimization.
   *
   * \param fixedImageCenterCoordinate The coordinate of the center point in the fixed image.
   * \param movingImagesPatchesJacobians A tensor to store the computed Jacobians (image gradients) for each patch.
   * \param patchIndex A vector defining the relative indices for extracting the patch.
   * \param patchSize The size of the patch to extract.
   * \param s An index to determine where to store the computed Jacobians in `movingImagesPatchesJacobians`.
   *
   * \return A tensor representing the extracted patch from the moving image.
   */
  torch::Tensor
  EvaluateMovingImagesPatchValuesAndJacobians(const FixedImagePointType &             fixedImageCenterCoordinate,
                                              torch::Tensor &                         movingImagesPatchesJacobians,
                                              const std::vector<std::vector<float>> & patchIndex,
                                              const std::vector<int64_t> &            patchSize,
                                              int                                     s) const;

  /**
   * \brief Generates valid patch indices and filters out invalid points.
   *
   * This method takes a list of fixed points and model configurations to generate patch indices
   * for each point. It filters out points that lie outside the mask or image boundaries, returning
   * only the valid fixed points. The valid patch indices are stored in `patchIndex`.
   *
   * \param modelConfig A vector of model configurations, each specifying the properties of the models used.
   * \param randomGenerator A random number generator used for random sampling in the patch generation process.
   * \param fixedPointsTmp A vector of fixed points to generate patch indices for.
   * \param patchIndex A reference to a 4D vector to store the generated patch indices for each valid fixed point.
   *
   * \return A vector of valid fixed points that are inside the mask or boundaries of the image.
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
  bool                      m_UseMixedPrecision;
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

  /**
   * \brief Retrieves a subset of features based on the provided indices.
   *
   * This method selects a subset of features from the full set of features, based on the indices
   * provided in `features_index`. It uses a random number generator for sampling a subset of the
   * features, ensuring that the selected features are randomly chosen.
   *
   * \param features_index A vector of indices representing the features to be considered.
   * \param randomGenerator A random number generator used for sampling the features.
   * \param n The number of features to select from the provided list of indices.
   *
   * \return A vector containing the indices of the randomly selected features.
   */
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

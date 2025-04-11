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

#ifndef elxImpactMetric_h
#define elxImpactMetric_h

#include "elxIncludes.h" // include first to avoid MSVS warning
#include "itkImpactImageToImageMetric.h"
#include "itkVector.h"

namespace elastix
{

/**
 * \class ImpactMetric
 * \brief A metric based on itk::ImpactImageToImageMetric.
 *
 * This metric compares semantic features extracted from one or more pretrained TorchScript models
 * applied to both the fixed and moving images. Feature vectors are compared using a configurable loss,
 * such as L1, L2, NCC, Cosine, or L1Cosine distance.
 *
 * ### Multi-model support
 * Multiple pretrained models can be specified for both the fixed and moving images. Each model can have its own
 * configuration (dimension, number of channels, patch size, voxel size, mask). To do this, simply provide
 * **space-separated values** for each parameter, in the same order as the models listed.
 *
 * Example:
 * \code
 * (ModelsPath "/Data/Models/TS/M850_8_Layers.pt /Data/Models/MIND/R1D2.pt")
 * (Dimension "3 3")
 * (NumberOfChannels "1 1")
 * (PatchSize "0*0*0 0*0*0")
 * (VoxelSize "1.5*1.5*1.5 6*6*6")
 * (LayersMask "00000001 1")
 * \endcode
 *
 * ### Multi-resolution support
 * All parameters support per-resolution configuration using Elastix's multi-resolution syntax.
 * For instance:
 * \code
 * (PatchSize "5*5*5 5*5*5" "3*3*3 3*3*3" "1*1*1 1*1*1")
 * \endcode
 * specifies different patch sizes for each resolution level and model.
 *
 * The same logic applies to:
 * - ModelsPath
 * - VoxelSize
 * - LayersMask
 * - SubsetFeatures
 * - LayersWeight
 * - PCA
 * - Distance
 *
 * If needed, you can provide separate configurations for the fixed and moving images
 * by adding the <tt>Fixed</tt> or <tt>Moving</tt> prefix to each parameter name.
 *
 * \code
 * (FixedModelsPath "...") and (MovingModelsPath "...")
 * \endcode
 *
 * All related parameters (e.g., <tt>Dimension</tt>, <tt>VoxelSize</tt>, <tt>LayersMask</tt>, etc.)
 * should then be specified separately using their <tt>Fixed*</tt> and <tt>Moving*</tt> versions.
 *
 * ### Key Parameters
 *
 * The parameters used in this class are:
 *
 * \param Mode Defines the operational mode of the metric. Possible values are:
 *   - "Static": features are precomputed and held fixed during optimization.
 *   - "Jacobian": gradients are backpropagated through the feature extractor.
 *
 * \param ModelsPath Specifies the path(s) to one or more TorchScript models used for feature extraction.
 *   Space-separated values allow combining multiple models in parallel.
 *   Example: <tt>(ModelsPath "/path/to/model1.pt /path/to/model2.pt")</tt>
 *
 * \param Dimension Defines the dimensionality of the input images (e.g., 2 or 3).
 *   This must match the input expectation of each model (one value per model).
 *   Example: <tt>(Dimension "3 2")</tt>
 *
 * \param NumberOfChannels Specifies the number of channels in the input images for each model.
 *   For grayscale, use 1. Example: <tt>(NumberOfChannels "1 3")</tt>
 *
 * \param PatchSize Size of the patch used for local feature extraction, given per model.
 *   Example: <tt>(PatchSize "5*5*5 7*7*7")</tt>
 *
 * \param VoxelSize Defines the physical spacing of voxels for each model's input space.
 *   This determines patch resolution. Example: <tt>(VoxelSize "1.5*1.5*1.5 3*3*3")</tt>
 *
 * \param LayersMask A binary string (per model) indicating which output layers of the model to use.
 *   Example: <tt>(LayersMask "00000001 1")</tt> uses the last layer of model 1 and layer 0 of model 2.
 *
 * \param SubsetFeatures Number of feature channels randomly selected per model.
 *   Example: <tt>(SubsetFeatures "1000 32")</tt>
 *
 * \param LayersWeight Relative importance of each selected model/layer in the total loss computation.
 *   Can be used to emphasize certain semantic levels. Example: <tt>(LayersWeight "1.0 0.5")</tt>
 *
 * \param Device Index of the GPU device to use. Set to -1 to force CPU execution.
 *   Example: <tt>(Device -1)</tt>
 *
 * \param PCA Number of principal components to retain per model during optional PCA-based feature compression.
 *   Set to 0 to disable PCA. Example: <tt>(PCA "32 3")</tt>
 *
 * \param Distance Specifies the similarity function to compare features.
 *   Supported values per model: <tt>L1</tt>, <tt>L2</tt>, <tt>NCC</tt>, <tt>Cosine</tt>, <tt>L1Cosine</tt>,
 * <tt>Dice</tt>. Example: <tt>(Distance "L2 Cosine")</tt>
 *
 * \param FeaturesMapUpdateInterval Frequency (in iterations) at which feature maps are recomputed in "Static" mode.
 *   Set to -1 to disable updates and keep features fixed. Example: <tt>(FeaturesMapUpdateInterval 10)</tt>
 *
 * \param WriteFeatureMaps Enables writing both the input images and the corresponding output feature maps
 *   to disk in "Static" mode, for each model and each resolution level. This is useful for inspection,
 *   debugging, or understanding which semantic features are being used during registration.
 *
 *   The files are saved to the output directory and follow the naming conventions:
 *
 *   - Input images:
 *     <tt>Fixed_<N>_<M>.mha</tt> and <tt>Moving_<N>_<M>.mha</tt>
 *     where <tt>N</tt> is the resolution level and <tt>M</tt> is the model index.
 *
 *   - Feature maps:
 *     <tt>FeatureMap/Fixed_<N>_<R1>_<R2>_<R3>.mha</tt> and <tt>Moving_<N>_<R1>_<R2>_<R3>.mha</tt>
 *     where <tt>R1, R2, R3</tt> are the voxels size.
 *
 *   Example: <tt>(WriteFeatureMaps "true")</tt>
 *
 *   Default is "false".
 *
 * \ingroup Metrics
 */

template <typename TElastix>
class ITK_TEMPLATE_EXPORT ImpactMetric
  : public itk::ImpactImageToImageMetric<typename MetricBase<TElastix>::FixedImageType,
                                         typename MetricBase<TElastix>::MovingImageType>
  , public MetricBase<TElastix>
{
public:
  ITK_DISALLOW_COPY_AND_MOVE(ImpactMetric);

  /** Standard ITK-stuff. */
  using Self = ImpactMetric;
  using Superclass1 = itk::ImpactImageToImageMetric<typename MetricBase<TElastix>::FixedImageType,
                                                    typename MetricBase<TElastix>::MovingImageType>;
  using Superclass2 = MetricBase<TElastix>;
  using Pointer = itk::SmartPointer<Self>;
  using ConstPointer = itk::SmartPointer<const Self>;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Run-time type information (and related methods). */
  itkOverrideGetNameOfClassMacro(ImpactMetric);

  /**
   * Name of this class.
   * Use this name in the parameter file to select this specific metric.
   * Example:
   * \code
   * (Metric "Impact")
   * \endcode
   */
  elxClassNameMacro("Impact");

  /** Typedefs from the superclass. */
  using typename Superclass1::CoordinateRepresentationType;
  using typename Superclass1::MovingImageType;
  using typename Superclass1::MovingImagePixelType;
  using typename Superclass1::MovingImageConstPointer;
  using typename Superclass1::FixedImageType;
  using typename Superclass1::FixedImageConstPointer;
  using typename Superclass1::FixedImageRegionType;
  using typename Superclass1::TransformType;
  using typename Superclass1::TransformPointer;
  using typename Superclass1::InputPointType;
  using typename Superclass1::OutputPointType;
  using typename Superclass1::TransformJacobianType;
  using typename Superclass1::InterpolatorType;
  using typename Superclass1::InterpolatorPointer;
  using typename Superclass1::RealType;
  using typename Superclass1::GradientPixelType;
  using typename Superclass1::GradientImageType;
  using typename Superclass1::GradientImagePointer;
  using typename Superclass1::FixedImageMaskType;
  using typename Superclass1::FixedImageMaskPointer;
  using typename Superclass1::MovingImageMaskType;
  using typename Superclass1::MovingImageMaskPointer;
  using typename Superclass1::MeasureType;
  using typename Superclass1::DerivativeType;
  using typename Superclass1::ParametersType;
  using typename Superclass1::FixedImagePixelType;
  using typename Superclass1::MovingImageRegionType;
  using typename Superclass1::ImageSamplerType;
  using typename Superclass1::ImageSamplerPointer;
  using typename Superclass1::ImageSampleContainerType;
  using typename Superclass1::ImageSampleContainerPointer;
  using typename Superclass1::FixedImageLimiterType;
  using typename Superclass1::MovingImageLimiterType;
  using typename Superclass1::FixedImageLimiterOutputType;
  using typename Superclass1::MovingImageLimiterOutputType;
  using typename Superclass1::MovingImageDerivativeScalesType;

  /** The fixed image dimension. */
  itkStaticConstMacro(FixedImageDimension, unsigned int, FixedImageType::ImageDimension);

  /** The moving image dimension. */
  itkStaticConstMacro(MovingImageDimension, unsigned int, MovingImageType::ImageDimension);

  /** Typedef's inherited from Elastix. */
  using typename Superclass2::ElastixType;
  using typename Superclass2::RegistrationType;
  using ITKBaseType = typename Superclass2::ITKBaseType;


  /** Sets up a timer to measure the initialization time and
   * calls the Superclass' implementation.
   */
  void
  Initialize() override;

  /** Update the current iteration and refresh feature maps in static mode.
   */
  void
  AfterEachIteration() override;

  /**
   * Do some things before each resolution:
   * \li Set CheckNumberOfSamples setting
   * \li Set UseNormalization setting
   */
  void
  BeforeEachResolution() override;


protected:
  /** The constructor. */
  ImpactMetric() = default;
  /** The destructor. */
  ~ImpactMetric() override = default;

  unsigned long m_CurrentIteration;

private:
  elxOverrideGetSelfMacro;

  /** Utility functions to parse Elastix-style vector strings. */
  template <typename T>
  std::vector<T>
  GetVectorFromString(int size, std::string valueStr, T defaultValue);
  template <typename T>
  std::vector<T>
  GetVectorFromString(int size, std::string valueStr, T defaultValue, char delimiter);
  template <typename T>
  std::vector<T>
  GetVectorFromString(std::string valueStr, T defaultValue, char delimiter);
  template <typename T>
  std::vector<T>
  GetVectorFromString(std::string valueStr, T defaultValue);

  std::vector<typename Superclass1::ModelConfiguration>
  GenerateModelsConfiguration(unsigned int level, std::string type, std::string mode, unsigned int imageDimension);
  template <typename T>
  std::string
  GetStringFromVector(const std::vector<T> & vec);
};

} // end namespace elastix

#ifndef ITK_MANUAL_INSTANTIATION
#  include "elxImpactMetric.hxx"
#endif

#endif // end #ifndef elxImpactMetric_h

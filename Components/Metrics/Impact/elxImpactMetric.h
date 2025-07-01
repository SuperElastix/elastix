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
#include "elxConfiguration.h"
#include "itkImpactImageToImageMetric.h"
#include "itkVector.h"

namespace elastix
{

/**
 * \class ImpactMetric
 * \brief A metric based on itk::ImpactImageToImageMetric.
 *
 * This metric compares semantic features extracted from one or more pretrained TorchScript models
 * applied to both the fixed and moving images. Feature vectors are compared using a configurable distance,
 * such as L1, L2, NCC, Cosine, Dice, or L1Cosine.
 *
 * ### Minimal Example
 * Example configuration for a single model and for one resolution 0:
 * \code
 * (ImpactModelsPath0 "/Data/Models/TS/M291_1_Layers.pt")
 * (ImpactDimension0 3)
 * (ImpactNumberOfChannels0 1)
 * (ImpactPatchSize0 5 5 5)
 * (ImpactVoxelSize0 1.5 1.5 1.5)
 * (ImpactLayersMask0 "1")
 * (ImpactSubsetFeatures0 32)
 * (ImpactPCA0 0)
 * (ImpactDistance0 "L2")
 * (ImpactLayersWeight0 1)
 * (ImpactMode "Jacobian")
 * (ImpactGPU 0)
 * (ImpactUseMixedPrecision "true")
 * (ImpactFeaturesMapUpdateInterval -1)
 * (ImpactWriteFeatureMaps "false")
 * \endcode
 *
 * ### Parameter Descriptions
 * The following parameters can be configured independently for each resolution level, using the corresponding index
 * (e.g., ImpactPatchSize0 for level 0, ImpactPatchSize1 for level 1, etc
 *
 * \param ImpactModelsPath Path to TorchScript model used for feature extraction.
 *
 * \param ImpactDimension Defines the dimensionality of the input expected by the TorchScript model.
 *   - `2` for 2D models (e.g., SAM2.1, DINOv2)
 *   - `3` for 3D models (e.g., TotalSegmentator, Anatomix, MIND).
 *
 * \param ImpactNumberOfChannels Specifies the number of channels in the input images for each model.
 *   - `1` for grayscale medical images (e.g., TotalSegmentator, Anatomix, MIND)
 *   - `3` for RGB-based models (e.g., SAM2.1, DINOv2)
 *
 * \param ImpactPatchSize The size of the patch used for feature extraction.
 *   - `X Y Z` for 3D models
 *   - `X Y` for 2D models
 *   If not all dimensions are specified, the first value is used to fill in missing dimensions (e.g., `5` becomes `5 5
 * 5` for 3D).
 *
 * \param ImpactVoxelSize Defines the physical spacing of the voxels (in millimeters).
 *   - `X Y Z` for 3D models
 *   - `X Y` for 2D models
 *   Use consistent voxel sizes to avoid inconsistencies during image pyramid processing.
 *
 * \param ImpactLayersMask Binary string indicating which output layers of the model to include in the similarity
 * computation. Example: `"00000001"` selects only the last layer.
 *
 * \param ImpactMode Defines how features are computed:
 *   - `"Static"`: Features are computed once per image and resolution level.
 *   - `"Jacobian"`: Features are computed at each iteration with backpropagation through the model.
 *
 * \param ImpactSubsetFeatures Number of feature channels randomly selected per voxel at each iteration.
 *
 * \param ImpactLayersWeight The relative importance of each layer in the similarity score.
 *
 * \param ImpactGPU Specifies the GPU device index, or `-1` for CPU execution.
 *
 * \param ImpactUseMixedPrecision Enables or disables the use of mixed precision (float16 and float32).
 *   Recommended to be disabled on CPU.
 *
 * \param ImpactPCA Number of principal components to retain for dimensionality reduction. Set to `0` to disable.
 *
 * \param ImpactDistance Specifies the distance metric to compare feature vectors. Supported values: `L1`, `L2`,
 * `Cosine`, `L1Cosine`, `Dice`, `NCC`, `DotProduct`.
 *
 * \param ImpactFeaturesMapUpdateInterval Controls how often feature maps are recomputed in "Static" mode.
 *   - Set to `-1` to compute once per resolution level.
 *   - Set to a positive integer to recompute every _N_ iterations.
 *
 * \param ImpactWriteFeatureMaps Enables saving both the input images and feature maps to disk (in Static mode).
 *
 * ### Advanced Use: Multi-resolution and Multi-model Setup
 *
 * IMPACT supports parallel use of multiple models and per-resolution customization. The following example illustrates
 * the configuration for two resolution levels: level 0 and level 1
 *
 * \code
 * (ImpactModelsPath0 "/Data/Models/TS/M291_1_Layers.pt")
 * (ImpactModelsPath1 "/Data/Models/SAM/Tiny_2_Layers.pt")
 * (ImpactDimension0 3)
 * (ImpactDimension1 2)
 * (ImpactNumberOfChannels0 1)
 * (ImpactNumberOfChannels1 3)
 * (ImpactPatchSize0 5 5 5)
 * (ImpactPatchSize1 29 29 29)
 * (ImpactVoxelSize0 3 3 3)
 * (ImpactVoxelSize1 1.5 1.5 1.5)
 * (ImpactLayersMask0 "1")
 * (ImpactLayersMask1 "01")
 * (ImpactPCA0 0)
 * (ImpactPCA1 3)
 * (ImpactSubsetFeatures0 32)
 * (ImpactSubsetFeatures1 3)
 * (ImpactDistance0 "L2")
 * (ImpactDistance1 "L1")
 * (ImpactLayersWeight0 1)
 * (ImpactLayersWeight1 1)
 * (ImpactMode "Static" "Jacobian")
 * (ImpactGPU 0 0)
 * (ImpactUseMixedPrecision "true" "true")
 * (ImpactFeaturesMapUpdateInterval -1 -1)
 * (ImpactWriteFeatureMaps "false" "false")
 * \endcode
 *
 * **Multi-model Setup**:
 * You can assign multiple models in parallel at a given resolution level by providing space-separated lists for each
 * parameter. In the following example, a 3D model with 8 output layers and a 3D MIND model are both used at resolution
 * level 0. Each entry in the lists corresponds to one of the two models
 * \code
 * (ImpactModelsPath0 "/Models/M850_8_Layers.pt" "/Models/MIND/R1D2.pt")
 * (ImpactDimension0 3 3)
 * (ImpactNumberOfChannels0 1 1)
 * (ImpactPatchSize0 5 5 5 7 7 7)
 * (ImpactVoxelSize0 1.5 1.5 1.5 6 6 6)
 * (ImpactLayersMask0 "00000001" "1")
 * (ImpactPCA0 0 0)
 * (ImpactSubsetFeatures0 64 16)
 * (ImpactDistance0 "Dice" "L2")
 * (ImpactLayersWeight0 1.0 0.5)
 * \endcode
 *
 * **Fixed and Moving-Specific Models**:
 * Assign different models to the fixed and moving images:
 * \code
 * (FixedModelsPath0 "/Models/TS/M850_8_Layers.pt")
 * (MovingModelsPath0 "/Models/MIND/R1D2.pt")
 * \endcode
 *
 * This allows asymmetric model configurations for different image modalities or anatomical content.
 *
 * \author V. Boussot,  Univ. Rennes, INSERM, LTSI- UMR 1099, F-35000 Rennes, France
 * \note This work was funded by the French National Research Agency as part of the VATSop project (ANR-20-CE19-0015).
 * \note If you use the Impact anywhere we would appreciate if you cite the following article:\n
 * V. Boussot et al., IMPACT: A Generic Semantic Loss for Multimodal Medical Image Registration, arXiv preprint
 * arXiv:2503.24121 (2025). https://doi.org/10.48550/arXiv.2503.24121
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

  std::vector<itk::ImpactModelConfiguration>
  GenerateModelsConfiguration(unsigned int level,
                              std::string  type,
                              std::string  mode,
                              unsigned int imageDimension,
                              bool         useMixedPrecision);
};

/**
 * \brief Parse a string into typed values using a custom delimiter.
 *
 * If `delimiter == '\0'`, characters are split one by one.
 * Otherwise, the string is split using the given delimiter.
 * Uses GetValueFromString<T>() for each token.
 *
 * \tparam T The type to parse.
 * \param valueStr The input string.
 * \param defaultValue The fallback value.
 * \param delimiter The token delimiter or `\0` for char-splitting.
 * \return A vector of parsed values.
 */
inline std::vector<bool>
GetBooleanVectorFromString(std::string valueStr, bool defaultValue)
{
  std::vector<bool> values(defaultValue, valueStr.size());
  for (char c : valueStr)
  {
    std::stringstream tokenStream(std::string(1, c));
    std::string       subToken;
    if (tokenStream >> subToken)
    {
      /** Cast the string to type bool. */
      bool value = subToken == "1";
      values.push_back(value);
    }
  }
  if (values.empty())
  {
    values.push_back(defaultValue);
  }
  return values;
} // end GetVectorFromString

/**
 * \brief Group a flattened space-separated string into string blocks per dimension sizes.
 *
 * This function splits a string like `"1 2 3 4 5"` into grouped strings according to
 * the values in `dimensions`. If dimensions = {2, 3}, result is {"1 2", "3 4 5"}.
 *
 * \param valueStr The space-separated values.
 * \param dimensions The sizes of each group.
 * \return A vector of grouped space-separated strings.
 */
template <typename T>
std::vector<std::vector<T>>
GroupByDimensions(const std::vector<T> & values, const std::vector<unsigned int> & dimensions)
{
  std::vector<std::vector<T>> grouped;
  size_t                      currentIndex = 0;
  size_t                      n = values.size();

  for (unsigned int dim : dimensions)
  {
    std::vector<T> group;
    for (unsigned int i = 0; i < dim; ++i)
    {
      if (currentIndex < n)
      {
        group.push_back(values[currentIndex++]);
      }
      else if (!values.empty())
      {
        group.push_back(values.back()); // répète la dernière valeur si débordement
      }
    }
    grouped.push_back(std::move(group));
  }

  return grouped;
}

} // end namespace elastix


#ifndef ITK_MANUAL_INSTANTIATION
#  include "elxImpactMetric.hxx"
#endif

#endif // end #ifndef elxImpactMetric_h

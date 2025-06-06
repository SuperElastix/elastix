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
 * Example configuration for a single model:
 * \code
 * (ImpactModelsPath "/Data/Models/TS/M291_1_Layers.pt")
 * (ImpactDimension 3)
 * (ImpactNumberOfChannels 1)
 * (ImpactPatchSize 5 5 5)
 * (ImpactVoxelSize 1.5 1.5 1.5)
 * (ImpactLayersMask "1")
 * (ImpactSubsetFeatures 32)
 * (ImpactPCA 0)
 * (ImpactDistance "L2")
 * (ImpactLayersWeight 1)
 * (ImpactMode "Jacobian")
 * (ImpactGPU 0)
 * (ImpactUseMixedPrecision "true")
 * (ImpactFeaturesMapUpdateInterval -1)
 * (ImpactWriteFeatureMaps "false")
 * \endcode
 *
 * ### Parameter Descriptions
 * The parameters used in this class are as follows:
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
 * IMPACT supports parallel use of multiple models and per-resolution customization. The following configurations are
 * supported:
 *
 * \code
 * (ImpactModelsPath "/Data/Models/TS/M291_1_Layers.pt" "/Data/Models/SAM/Tiny_2_Layers.pt")
 * (ImpactDimension 3 2)
 * (ImpactNumberOfChannels 1 3)
 * (ImpactPatchSize 5 5 5 29 29 29)
 * (ImpactVoxelSize 3 3 3 1.5 1.5 1.5)
 * (ImpactLayersMask "1" "01")
 * (ImpactPCA 0 3)
 * (ImpactSubsetFeatures 32 3)
 * (ImpactDistance "L2" "L1")
 * (ImpactLayersWeight 1 1)
 * (ImpactMode "Static" "Jacobian")
 * (ImpactGPU 0 0)
 * (ImpactUseMixedPrecision "true" "true")
 * (ImpactFeaturesMapUpdateInterval -1 -1)
 * (ImpactWriteFeatureMaps "false" "false")
 * \endcode
 *
 * **Multi-model Setup**:
 * Use space-separated lists for different models:
 * \code
 * (ImpactModelsPath "/Models/M850_8_Layers.pt /Models/MIND/R1D2.pt")
 * (ImpactDimension "3 3")
 * (ImpactNumberOfChannels "1 1")
 * (ImpactPatchSize "5 5 5 7 7 7")
 * (ImpactVoxelSize "1.5 1.5 1.5 6 6 6")
 * (ImpactLayersMask "00000001 1")
 * (ImpactPCA "0 0")
 * (ImpactSubsetFeatures "64 16")
 * (ImpactDistance "Dice L2")
 * (ImpactLayersWeight "1.0 0.5")
 * \endcode
 *
 * **Fixed and Moving-Specific Models**:
 * Assign different models to the fixed and moving images:
 * \code
 * (FixedModelsPath "/Models/TS/M850_8_Layers.pt")
 * (MovingModelsPath "/Models/MIND/R1D2.pt")
 * \endcode
 *
 * This allows asymmetric model configurations for different image modalities or anatomical content.
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
 * \brief Convert a string token to a typed value with bounds checking.
 *
 * This function parses a string and converts it to the specified type `T`.
 * For booleans, only "1" and "0" are accepted. For numeric types, the value
 * is checked against type limits, and exceptions are thrown if the value
 * is out of range or the conversion fails.
 *
 * \tparam T The type to convert to.
 * \param token A string representing a value of type T.
 * \return The parsed value.
 *
 * \exception itk::ExceptionObject if parsing fails or value is out of bounds.
 */
template <typename T>
T
GetValueFromString(std::string token)
{
  T value;
  if constexpr (std::is_same_v<T, std::string>)
  {
    value = token;
  }
  else if constexpr (std::is_same_v<T, bool>)
  {
    if (token == "1")
    {
      value = true;
    }
    else if (token == "0")
    {
      value = false;
    }
    else
    {
      itkGenericExceptionMacro("Invalid boolean string: '" << token << "'. Expected '1'/'0'.");
    }
  }
  else
  {
    std::stringstream tokenStream(token);
    long double       parsedValue;
    if (tokenStream >> parsedValue && tokenStream.eof())
    {
      if constexpr (std::is_integral_v<T>)
      {
        if (parsedValue < static_cast<long double>(std::numeric_limits<T>::min()))
        {
          itkGenericExceptionMacro("Value '" << parsedValue << "' is below min value of type " << typeid(T).name());
        }
        if (parsedValue > static_cast<long double>(std::numeric_limits<T>::max()))
        {
          itkGenericExceptionMacro("Value '" << parsedValue << "' exceeds max value of type " << typeid(T).name());
        }
      }
      else if constexpr (std::is_floating_point_v<T>)
      {
        if (parsedValue < -std::numeric_limits<T>::max())
        {
          itkGenericExceptionMacro("Value '" << parsedValue << "' is below -max of float type " << typeid(T).name());
        }
        if (parsedValue > std::numeric_limits<T>::max())
        {
          itkGenericExceptionMacro("Value '" << parsedValue << "' exceeds max float value of type "
                                             << typeid(T).name());
        }
      }
      value = static_cast<T>(parsedValue);
    }
    else
    {
      itkGenericExceptionMacro("Could not parse token '" << token << "' as type " << typeid(T).name());
    }
  }
  return value;
} // end GetValueFromString

/**
 * \brief Convert a space-delimited string into a vector of typed values.
 *
 * Each token is parsed using GetValueFromString<T>(). If the string is empty,
 * the default value is returned in a single-element vector.
 *
 * \tparam T The type to parse.
 * \param valueStr The input string.
 * \param defaultValue The fallback value if input is empty.
 * \return A vector of parsed values.
 */
template <typename T>
std::vector<T>
GetVectorFromString(std::string valueStr, T defaultValue)
{
  std::stringstream ss(valueStr);
  std::vector<T>    values;
  std::string       token;

  while (ss >> token)
  {
    values.push_back(GetValueFromString<T>(token));
  }

  if (values.empty())
  {
    values.push_back(defaultValue);
  }
  return values;
} // end GetVectorFromString

/**
 * \brief Parse a space-delimited string into a vector of a fixed size.
 *
 * Each token is parsed using GetValueFromString<T>().
 * If not enough values are provided, the first value is duplicated.
 * If too many values are provided, the result is truncated.
 *
 * \tparam T The type to parse.
 * \param size Target size of the output vector.
 * \param valueStr The input string.
 * \param defaultValue Fallback if parsing fails.
 * \return A fixed-size vector of parsed values.
 */
template <typename T>
std::vector<T>
GetVectorFromString(int size, std::string valueStr, T defaultValue)
{
  std::vector<T> values = GetVectorFromString<T>(valueStr, defaultValue);
  if (values.size() > size)
  {
    values.resize(size);
  }
  while (values.size() < size)
  {
    values.push_back(values[0]);
  }
  return values;
} // end GetVectorFromString


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
  std::vector<bool> values;
  for (char c : valueStr)
  {
    std::stringstream tokenStream(std::string(1, c));
    std::string       subToken;
    if (tokenStream >> subToken)
    {
      values.push_back(GetValueFromString<bool>(subToken));
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
inline std::vector<std::string>
groupStrByDimensions(std::string valueStr, std::vector<unsigned int> dimensions)
{
  std::stringstream        ss(valueStr);
  std::vector<std::string> flatValues;
  std::string              value;

  while (ss >> value)
  {
    flatValues.push_back(value);
  }

  std::vector<std::string> groups;
  size_t                   index = 0;

  for (unsigned int & dim : dimensions)
  {
    std::ostringstream groupStream;
    for (unsigned int i = 0; i < dim && index < flatValues.size(); ++i, ++index)
    {
      if (i > 0)
        groupStream << " ";
      groupStream << flatValues[index];
    }
    groups.push_back(groupStream.str());
  }

  return groups;
}

/**
 * \brief Format parameter values by level and model dimension (Jacobian mode).
 *
 * This version infers model dimension per level from the parameter `<prefix>Dimension`.
 * The output collects parameters for the requested level. If a value at that level
 * contains multiple space-separated entries, it is returned directly.
 *
 * \param config Pointer to configuration object.
 * \param prefix Parameter prefix (e.g., "Impact").
 * \param parameterName Name of the parameter suffix (e.g., "PatchSize").
 * \param level Level index to extract values from.
 * \return Formatted string containing values.
 */
inline std::string
formatParameterStringByDimensionAndLevel(const Configuration * config,
                                         const std::string &   prefix,
                                         const std::string &   parameterName,
                                         int                   level)
{
  std::ostringstream paramStream;
  std::ostringstream paramDefault;
  int                paramIndex = 0;
  bool               stop = false;

  for (int l = 0; l <= level && !stop; ++l)
  {
    std::string modelDimension;
    config->ReadParameter(modelDimension, prefix + "Dimension", l, 0);
    std::vector<unsigned int> modelsDimensionVec = GetVectorFromString<unsigned int>(1, modelDimension, 3);

    for (unsigned int d = 0; d < modelsDimensionVec[0] && !stop; ++d)
    {
      std::string paramStrTmp;
      bool        hasParam = config->ReadParameter(paramStrTmp, prefix + parameterName, paramIndex, 0);

      if (!hasParam)
      {
        stop = true;
        break;
      }
      if (l == level)
      {
        if (paramStrTmp.find(" ") == std::string::npos)
        {
          paramStream << (paramStream.str().empty() ? paramStrTmp : " " + paramStrTmp);
        }
        else
        {
          if (paramStream.str().empty())
          {
            paramStream << paramStrTmp;
          }
          stop = true;
        }
      }
      ++paramIndex;
      if (paramStrTmp.find(" ") != std::string::npos)
      {
        break;
      }
    }
  }

  return paramStream.str();
} // end formatParameterStringByDimensionAndLevel

/**
 * \brief Format parameter values by level with a fixed image dimension (static mode).
 *
 * Similar to the above function but uses a constant dimension instead of a per-level model dimension.
 *
 * \param config Pointer to configuration object.
 * \param prefix Parameter prefix (e.g., "Impact").
 * \param parameterName Name of the parameter (e.g., "PatchSize").
 * \param level Level index to extract values from.
 * \param dimension Fixed dimension used to select parameters.
 * \return Formatted string containing values.
 */
inline std::string
formatParameterStringByDimensionAndLevel(const Configuration * config,
                                         const std::string &   prefix,
                                         const std::string &   parameterName,
                                         int                   level,
                                         unsigned int          dimension)
{
  std::ostringstream paramStream;
  int                paramIndex = 0;
  bool               stop = false;

  for (int l = 0; l <= level && !stop; ++l)
  {
    for (unsigned int d = 0; d < dimension && !stop; ++d)
    {
      std::string paramStrTmp;
      bool        hasParam = config->ReadParameter(paramStrTmp, prefix + parameterName, paramIndex, 0);

      if (!hasParam)
      {
        stop = true;
        break;
      }

      if (l == level)
      {
        if (paramStrTmp.find(" ") == std::string::npos)
        {
          paramStream << (paramStream.str().empty() ? paramStrTmp : " " + paramStrTmp);
        }
        else
        {
          if (paramStream.str().empty())
          {
            paramStream << paramStrTmp;
          }
          stop = true;
        }
      }
      ++paramIndex;
      if (paramStrTmp.find(" ") != std::string::npos)
      {
        break;
      }
    }
  }

  return paramStream.str();
} // end formatParameterStringByDimensionAndLevel

} // end namespace elastix


#ifndef ITK_MANUAL_INSTANTIATION
#  include "elxImpactMetric.hxx"
#endif

#endif // end #ifndef elxImpactMetric_h

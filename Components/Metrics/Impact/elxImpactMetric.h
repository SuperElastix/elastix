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
 * (ImpactModelsPath "/Data/Models/TS/M850_8_Layers.pt")
 * (ImpactDimension 3)
 * (ImpactNumberOfChannels 1)
 * (ImpactPatchSize 0 0 0)
 * (ImpactVoxelSize 1.5 1.5 1.5)
 * (ImpactLayersMask "10000001")
 * (ImpactPCA 3)
 * (ImpactSubsetFeatures 3 10)
 * (ImpactLayersWeight 0.5 0.5)
 * (ImpactDistance "L2")
 * \endcode
 *
 * ### Multi-resolution support
 * All parameters support per-resolution configuration using Elastix's multi-resolution syntax.
 * For instance for a 2 resolution setup:
 * \code
 * (ImpactModelsPath "/Data/Models/TS/M850_8_Layers.pt" "/Data/Models/SAM/Tiny_2_Layers.pt")
 * (ImpactDimension 3 2)
 * (ImpactNumberOfChannels 1 3)
 * (ImpactPatchSize 5 5 5 29 29 29)
 * (ImpactVoxelSize 3 3 3 1.5 1.5 1.5)
 * (ImpactLayersMask "00000001" "01")
 * \endcode

 * If needed, you can provide separate configurations for the fixed and moving images
 * by adding the <tt>ImpactFixed</tt> or <tt>ImpactMoving</tt> prefix to each parameter name.
 *
 * \code
 * (ImpactFixedModelsPath "...") and (ImpactMovingModelsPath "...")
 * \endcode
 *
 * All related parameters (e.g., <tt>Dimension</tt>, <tt>VoxelSize</tt>, <tt>LayersMask</tt>, etc.)
 * should then be specified separately using their <tt>ImpactFixed*</tt> and <tt>ImpactMoving*</tt> versions.
 *
 * ### Key Parameters
 *
 * The parameters used in this class are:
 *
 * \param ImpactMode Defines the operational mode of the metric. Possible values are:
 *   - "Static": features are precomputed and held fixed during optimization.
 *   - "Jacobian": gradients are backpropagated through the feature extractor.
 *
 * \param ImpactModelsPath Specifies the path(s) to one or more TorchScript models used for feature extraction.
 *   Space-separated values allow combining multiple models in parallel.
 *   Example: <tt>(ImpactModelsPath "/path/to/model1.pt /path/to/model2.pt")</tt>
 *
 * \param ImpactDimension Defines the dimensionality of the input images (e.g., 2 or 3).
 *   This must match the input expectation of each model (one value per model).
 *   Example: <tt>(ImpactDimension "3 2")</tt>
 *
 * \param ImpactNumberOfChannels Specifies the number of channels in the input images for each model.
 *   For grayscale, use 1. Example: <tt>(ImpactNumberOfChannels "1 3")</tt>
 *
 * \param ImpactPatchSize Size of the patch used for local feature extraction, given per model.
 *   Example: <tt>(ImpactPatchSize "5 5 5 7 7 7")</tt>
 *
 * \param ImpactVoxelSize Defines the physical spacing of voxels for each model's input space.
 *   This determines patch resolution. Example: <tt>(ImpactVoxelSize "1.5 1.5 1.5 3 3 3")</tt>
 *
 * \param ImpactLayersMask A binary string (per model) indicating which output layers of the model to use.
 *   Example: <tt>(ImpactLayersMask "00000001 1")</tt> uses the last layer of model 1 and layer 0 of model 2.
 *
 * \param ImpactSubsetFeatures Number of feature channels randomly selected per model.
 *   Example: <tt>(ImpactSubsetFeatures "1000 32")</tt>
 *
 * \param ImpactLayersWeight Relative importance of each selected model/layer in the total loss computation.
 *   Can be used to emphasize certain semantic levels. Example: <tt>(ImpactLayersWeight "1.0 0.5")</tt>
 *
 * \param GOU Index of the GPU device to use. Set to -1 to force CPU execution.
 *   Example: <tt>(ImpactGPU -1)</tt>
 *
 * \param ImpactPCA Number of principal components to retain per model during optional PCA-based feature compression.
 *   Set to 0 to disable PCA. Example: <tt>(ImpactPCA "32 3")</tt>
 *
 * \param ImpactDistance Specifies the similarity function to compare features.
 *   Supported values per model: <tt>L1</tt>, <tt>L2</tt>, <tt>NCC</tt>, <tt>Cosine</tt>, <tt>L1Cosine</tt>,
 * <tt>Dice</tt>. Example: <tt>(ImpactDistance "L2 Cosine")</tt>
 *
 * \param ImpactFeaturesMapUpdateInterval Frequency (in iterations) at which feature maps are recomputed in "Static"
 * mode. Set to -1 to disable updates and keep features fixed. Example: <tt>(ImpactFeaturesMapUpdateInterval 10)</tt>
 *
 * \param ImpactWriteFeatureMaps Enables writing both the input images and the corresponding output feature maps
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
 *   Example: <tt>(ImpactWriteFeatureMaps "true")</tt>
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

  std::vector<typename Superclass1::ModelConfiguration>
  GenerateModelsConfiguration(unsigned int level, std::string type, std::string mode, unsigned int imageDimension);
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

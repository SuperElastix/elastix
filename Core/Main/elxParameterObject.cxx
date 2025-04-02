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

#include "elxParameterObject.h"
#include "elxConversion.h"

#include "itkParameterFileParser.h"

#include "itkFileTools.h"
#include <fstream>
#include <iostream>
#include <cmath>

namespace elastix
{

/**
 * ********************* SetParameterMap *********************
 */

void
ParameterObject::SetParameterMap(const ParameterMapType & parameterMap)
{
  ParameterMapVectorType parameterMapVector(1, parameterMap);
  this->SetParameterMaps(parameterMapVector);
}

/**
 * ********************* SetParameterMap *********************
 */

void
ParameterObject::SetParameterMap(const unsigned int index, const ParameterMapType & parameterMap)
{
  m_ParameterMaps[index] = parameterMap;
}


/**
 * ********************* SetParameterMap *********************
 */

void
ParameterObject::SetParameterMap(const ParameterMapVectorType & parameterMaps)
{
  if (m_ParameterMaps != parameterMaps)
  {
    m_ParameterMaps = parameterMaps;
    this->Modified();
  }
}


/**
 * ********************* SetParameterMaps *********************
 */

void
ParameterObject::SetParameterMaps(const ParameterMapVectorType & parameterMaps)
{
  if (m_ParameterMaps != parameterMaps)
  {
    m_ParameterMaps = parameterMaps;
    this->Modified();
  }
}


/**
 * ********************* AddParameterMap *********************
 */

void
ParameterObject::AddParameterMap(const ParameterMapType & parameterMap)
{
  m_ParameterMaps.push_back(parameterMap);
  this->Modified();
}


/**
 * ********************* GetParameterMap *********************
 */

const ParameterObject::ParameterMapType &
ParameterObject::GetParameterMap(const unsigned int index) const
{
  return m_ParameterMaps[index];
}


/**
 * ********************* SetParameter *********************
 */

void
ParameterObject::SetParameter(const unsigned int index, const ParameterKeyType & key, const ParameterValueType & value)
{
  m_ParameterMaps[index][key] = ParameterValueVectorType(1, value);
}


/**
 * ********************* SetParameter *********************
 */

void
ParameterObject::SetParameter(const unsigned int               index,
                              const ParameterKeyType &         key,
                              const ParameterValueVectorType & value)
{
  m_ParameterMaps[index][key] = value;
}


/**
 * ********************* SetParameter *********************
 */

void
ParameterObject::SetParameter(const ParameterKeyType & key, const ParameterValueType & value)
{
  for (auto & parameterMap : m_ParameterMaps)
  {
    parameterMap[key] = ParameterValueVectorType(1, value);
  }
}


/**
 * ********************* SetParameter *********************
 */

void
ParameterObject::SetParameter(const ParameterKeyType & key, const ParameterValueVectorType & value)
{
  for (auto & parameterMap : m_ParameterMaps)
  {
    parameterMap[key] = value;
  }
}


/**
 * ********************* GetParameter *********************
 */

const ParameterObject::ParameterValueVectorType &
ParameterObject::GetParameter(const unsigned int index, const ParameterKeyType & key)
{
  return m_ParameterMaps[index][key];
}


/**
 * ********************* RemoveParameter *********************
 */

void
ParameterObject::RemoveParameter(const unsigned int index, const ParameterKeyType & key)
{
  m_ParameterMaps[index].erase(key);
}


/**
 * ********************* RemoveParameter *********************
 */

void
ParameterObject::RemoveParameter(const ParameterKeyType & key)
{
  for (unsigned int index = 0; index < this->GetNumberOfParameterMaps(); ++index)
  {
    this->RemoveParameter(index, key);
  }
}


/**
 * ********************* ReadParameterFile *********************
 */

void
ParameterObject::ReadParameterFile(const ParameterFileNameType & parameterFileName)
{
  this->SetParameterMaps({ itk::ParameterFileParser::ReadParameterMap(parameterFileName) });
}


// Deprecated, superseded by ReadParameterFiles.
void
ParameterObject::ReadParameterFile(const ParameterFileNameVectorType & parameterFileNameVector)
{
  this->ReadParameterFiles(parameterFileNameVector);
}


/**
 * ********************* ReadParameterFiles *********************
 */

void
ParameterObject::ReadParameterFiles(const ParameterFileNameVectorType & parameterFileNameVector)
{
  if (parameterFileNameVector.empty())
  {
    itkExceptionMacro("Parameter filename container is empty.");
  }

  m_ParameterMaps.clear();

  for (const auto & parameterFileName : parameterFileNameVector)
  {
    if (!itksys::SystemTools::FileExists(parameterFileName))
    {
      itkExceptionMacro("Parameter file \"" << parameterFileName << "\" does not exist.");
    }

    this->AddParameterFile(parameterFileName);
  }
}


/**
 * ********************* AddParameterFile *********************
 */

void
ParameterObject::AddParameterFile(const ParameterFileNameType & parameterFileName)
{
  m_ParameterMaps.push_back(itk::ParameterFileParser::ReadParameterMap(parameterFileName));
}


/**
 * ********************* WriteParameterFile *********************
 */


// Deprecated, superseded by WriteParameterFiles.
void
ParameterObject::WriteParameterFile() const
{
  this->WriteParameterFiles();
}


void
ParameterObject::WriteParameterFile(const ParameterMapType &      parameterMap,
                                    const ParameterFileNameType & parameterFileName)
{
  std::ofstream parameterFile;
  parameterFile.exceptions(std::ofstream::failbit | std::ofstream::badbit);
  parameterFile << std::fixed;

  try
  {
    parameterFile.open(parameterFileName, std::ofstream::out);
  }
  catch (const std::ios_base::failure & e)
  {
    itkGenericExceptionMacro("Error opening parameter file: " << e.what());
  }

  try
  {
    const auto format = itksys::SystemTools::GetFilenameLastExtension(parameterFileName) ==
                            Conversion::CreateParameterMapFileNameExtension(ParameterMapStringFormat::Toml)
                          ? ParameterMapStringFormat::Toml
                          : ParameterMapStringFormat::LegacyTxt;

    parameterFile << Conversion::ParameterMapToString(parameterMap, format);
  }
  catch (const std::ios_base::failure & e)
  {
    itkGenericExceptionMacro("Error writing to paramter file: " << e.what());
  }

  try
  {
    parameterFile.close();
  }
  catch (const std::ios_base::failure & e)
  {
    itkGenericExceptionMacro("Error closing parameter file:" << e.what());
  }
}


void
ParameterObject::WriteParameterFile(const ParameterFileNameType & parameterFileName) const
{
  if (m_ParameterMaps.empty())
  {
    itkExceptionMacro("Error writing parameter map to disk: The parameter object is empty.");
  }

  if (m_ParameterMaps.size() > 1)
  {
    itkExceptionMacro("Error writing to disk: The number of parameter maps ("
                      << m_ParameterMaps.size()
                      << ") does not match the number of provided filenames (1). Please call WriteParameterFiles "
                         "instead, and provide a vector of filenames.");
  }

  this->WriteParameterFile(m_ParameterMaps[0], parameterFileName);
}


// Deprecated, superseded by WriteParameterFiles.
void
ParameterObject::WriteParameterFile(const ParameterMapVectorType &      parameterMapVector,
                                    const ParameterFileNameVectorType & parameterFileNameVector)
{
  Self::WriteParameterFiles(parameterMapVector, parameterFileNameVector);
}


// Deprecated, superseded by WriteParameterFiles.
void
ParameterObject::WriteParameterFile(const ParameterFileNameVectorType & parameterFileNameVector) const
{
  this->WriteParameterFiles(parameterFileNameVector);
}


/**
 * ********************* WriteParameterFiles *********************
 */


void
ParameterObject::WriteParameterFiles() const
{
  ParameterFileNameVectorType parameterFileNameVector;
  for (unsigned int i = 0; i < m_ParameterMaps.size(); ++i)
  {
    parameterFileNameVector.push_back("ParametersFile." + std::to_string(i) + ".txt");
  }

  Self::WriteParameterFiles(m_ParameterMaps, parameterFileNameVector);
}


void
ParameterObject::WriteParameterFiles(const ParameterMapVectorType &      parameterMapVector,
                                     const ParameterFileNameVectorType & parameterFileNameVector)
{
  if (parameterMapVector.size() != parameterFileNameVector.size())
  {
    itkGenericExceptionMacro("Error writing to disk: The number of parameter maps ("
                             << parameterMapVector.size() << ") does not match the number of provided filenames ("
                             << parameterFileNameVector.size() << ").");
  }

  if (!parameterMapVector.empty())
  {
    Self::WriteParameterFile(parameterMapVector.front(), parameterFileNameVector.front());
  }

  // Add initial transform parameter file names. Do not touch the first one,
  // since it may have one already
  for (unsigned int i = 1; i < parameterMapVector.size(); ++i)
  {
    ParameterMapType parameterMap = parameterMapVector[i];
    if (parameterMap.find("TransformParameters") != parameterMap.end())
    {
      parameterMap["InitialTransformParameterFileName"][0] = parameterFileNameVector[i - 1];
    }

    Self::WriteParameterFile(parameterMap, parameterFileNameVector[i]);
  }
}


void
ParameterObject::WriteParameterFiles(const ParameterFileNameVectorType & parameterFileNameVector) const
{
  Self::WriteParameterFiles(m_ParameterMaps, parameterFileNameVector);
}


/**
 * ********************* GetDefaultParameterMap *********************
 */

ParameterObject::ParameterMapType
ParameterObject::GetDefaultParameterMap(const std::string & transformName,
                                        const unsigned int  numberOfResolutions,
                                        const double        finalGridSpacingInPhysicalUnits)
{
  // Parameters that depend on size and number of resolutions
  ParameterMapType parameterMap{};

  // Common Components
  parameterMap["FixedImagePyramid"] = ParameterValueVectorType(1, "FixedSmoothingImagePyramid");
  parameterMap["MovingImagePyramid"] = ParameterValueVectorType(1, "MovingSmoothingImagePyramid");
  parameterMap["Interpolator"] = ParameterValueVectorType(1, "LinearInterpolator");
  parameterMap["Optimizer"] = ParameterValueVectorType(1, "AdaptiveStochasticGradientDescent");
  parameterMap["Resampler"] = ParameterValueVectorType(1, "DefaultResampler");
  parameterMap["ResampleInterpolator"] = ParameterValueVectorType(1, "FinalBSplineInterpolator");
  parameterMap["FinalBSplineInterpolationOrder"] = ParameterValueVectorType(1, "3");
  parameterMap["NumberOfResolutions"] = ParameterValueVectorType(1, std::to_string(numberOfResolutions));
  parameterMap["WriteIterationInfo"] = ParameterValueVectorType(1, "false");

  // Image Sampler
  parameterMap["ImageSampler"] = ParameterValueVectorType(1, "RandomCoordinate");
  parameterMap["NumberOfSpatialSamples"] = ParameterValueVectorType(1, "2048");
  parameterMap["CheckNumberOfSamples"] = ParameterValueVectorType(1, "true");
  parameterMap["MaximumNumberOfSamplingAttempts"] = ParameterValueVectorType(1, "8");
  parameterMap["NewSamplesEveryIteration"] = ParameterValueVectorType(1, "true");

  // Optimizer
  parameterMap["NumberOfSamplesForExactGradient"] = ParameterValueVectorType(1, "4096");
  parameterMap["DefaultPixelValue"] = ParameterValueVectorType(1, "0");
  parameterMap["AutomaticParameterEstimation"] = ParameterValueVectorType(1, "true");

  // Output
  parameterMap["WriteResultImage"] = ParameterValueVectorType(1, "true");
  parameterMap["ResultImageFormat"] = ParameterValueVectorType(1, "nii");

  // transformNames
  if (transformName == "translation")
  {
    parameterMap["Registration"] = ParameterValueVectorType(1, "MultiResolutionRegistration");
    parameterMap["Transform"] = ParameterValueVectorType(1, "TranslationTransform");
    parameterMap["Metric"] = ParameterValueVectorType(1, "AdvancedMattesMutualInformation");
    parameterMap["MaximumNumberOfIterations"] = ParameterValueVectorType(1, "256");
    parameterMap["AutomaticTransformInitialization"] = ParameterValueVectorType(1, "true");
  }
  else if (transformName == "rigid")
  {
    parameterMap["Registration"] = ParameterValueVectorType(1, "MultiResolutionRegistration");
    parameterMap["Transform"] = ParameterValueVectorType(1, "EulerTransform");
    parameterMap["Metric"] = ParameterValueVectorType(1, "AdvancedMattesMutualInformation");
    parameterMap["MaximumNumberOfIterations"] = ParameterValueVectorType(1, "256");
    parameterMap["AutomaticScalesEstimation"] = ParameterValueVectorType(1, "true");
  }
  else if (transformName == "affine")
  {
    parameterMap["Registration"] = ParameterValueVectorType(1, "MultiResolutionRegistration");
    parameterMap["Transform"] = ParameterValueVectorType(1, "AffineTransform");
    parameterMap["Metric"] = ParameterValueVectorType(1, "AdvancedMattesMutualInformation");
    parameterMap["MaximumNumberOfIterations"] = ParameterValueVectorType(1, "256");
    parameterMap["AutomaticScalesEstimation"] = ParameterValueVectorType(1, "true");
  }
  else if (transformName == "bspline" || transformName == "nonrigid") // <-- nonrigid for backwards compatibility
  {
    parameterMap["Registration"] = ParameterValueVectorType(1, "MultiMetricMultiResolutionRegistration");
    parameterMap["Transform"] = ParameterValueVectorType(1, "BSplineTransform");
    parameterMap["Metric"] = ParameterValueVectorType(1, "AdvancedMattesMutualInformation");
    parameterMap["Metric"].push_back("TransformBendingEnergyPenalty");
    parameterMap["Metric0Weight"] = ParameterValueVectorType(1, "1.0");
    parameterMap["Metric1Weight"] = ParameterValueVectorType(1, "1.0");
    parameterMap["MaximumNumberOfIterations"] = ParameterValueVectorType(1, "256");
  }
  else if (transformName == "spline")
  {
    parameterMap["Registration"] = ParameterValueVectorType(1, "MultiResolutionRegistration");
    parameterMap["Transform"] = ParameterValueVectorType(1, "SplineKernelTransform");
    parameterMap["Metric"] = ParameterValueVectorType(1, "AdvancedMattesMutualInformation");
    parameterMap["MaximumNumberOfIterations"] = ParameterValueVectorType(1, "256");
  }
  else if (transformName == "groupwise")
  {
    parameterMap["Registration"] = ParameterValueVectorType(1, "MultiResolutionRegistration");
    parameterMap["Transform"] = ParameterValueVectorType(1, "BSplineStackTransform");
    parameterMap["Metric"] = ParameterValueVectorType(1, "VarianceOverLastDimensionMetric");
    parameterMap["MaximumNumberOfIterations"] = ParameterValueVectorType(1, "256");
    parameterMap["Interpolator"] = ParameterValueVectorType(1, "ReducedDimensionBSplineInterpolator");
    parameterMap["ResampleInterpolator"] = ParameterValueVectorType(1, "FinalReducedDimensionBSplineInterpolator");
  }
  else
  {
    itkGenericExceptionMacro("No default parameter map \"" << transformName << "\".");
  }

  // B-spline transform settings
  if (transformName == "bspline" || transformName == "nonrigid" ||
      transformName == "groupwise") // <-- nonrigid for backwards compatibility
  {
    ParameterValueVectorType gridSpacingSchedule{};
    for (double resolution = 0; resolution < numberOfResolutions; ++resolution)
    {
      gridSpacingSchedule.insert(gridSpacingSchedule.begin(), std::to_string(std::pow(1.41, resolution)));
    }

    parameterMap["GridSpacingSchedule"] = gridSpacingSchedule;
    parameterMap["FinalGridSpacingInPhysicalUnits"] =
      ParameterValueVectorType(1, std::to_string(finalGridSpacingInPhysicalUnits));
  }

  return parameterMap;
}


/**
 * ********************* PrintSelf *********************
 */

void
ParameterObject::PrintSelf(std::ostream & os, itk::Indent indent) const
{
  Superclass::PrintSelf(os, indent);

  for (unsigned int i = 0; i < m_ParameterMaps.size(); ++i)
  {
    os << "ParameterMap " << i << ": " << std::endl;
    auto parameterMapIterator = m_ParameterMaps[i].begin();
    auto parameterMapIteratorEnd = m_ParameterMaps[i].end();
    while (parameterMapIterator != parameterMapIteratorEnd)
    {
      os << "  (" << parameterMapIterator->first;
      ParameterValueVectorType parameterMapValueVector = parameterMapIterator->second;

      for (const std::string & value : parameterMapValueVector)
      {
        std::istringstream stream(value);
        float              number;
        stream >> number;
        if (stream.fail())
        {
          os << " \"" << value << '"';
        }
        else
        {
          os << ' ' << number;
        }
      }

      os << ')' << std::endl;
      ++parameterMapIterator;
    }
  }
}


} // namespace elastix

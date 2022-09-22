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
  ParameterMapVectorType parameterMapVector = ParameterMapVectorType(1, parameterMap);
  this->SetParameterMap(parameterMapVector);
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
  this->SetParameterMap(ParameterMapVectorType{ itk::ParameterFileParser::ReadParameterMap(parameterFileName) });
}


/**
 * ********************* ReadParameterFile *********************
 */

void
ParameterObject::ReadParameterFile(const ParameterFileNameVectorType & parameterFileNameVector)
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
      itkExceptionMacro("Parameter file \"" << parameterFileName << "\" does not exist.")
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


void
ParameterObject::WriteParameterFile() const
{
  ParameterFileNameVectorType parameterFileNameVector;
  for (unsigned int i = 0; i < m_ParameterMaps.size(); ++i)
  {
    parameterFileNameVector.push_back("ParametersFile." + std::to_string(i) + ".txt");
  }

  Self::WriteParameterFile(m_ParameterMaps, parameterFileNameVector);
}


/**
 * ********************* WriteParameterFile *********************
 */

void
ParameterObject::WriteParameterFile(const ParameterMapType &      parameterMap,
                                    const ParameterFileNameType & parameterFileName)
{
  std::ofstream parameterFile;
  parameterFile.exceptions(std::ofstream::failbit | std::ofstream::badbit);
  parameterFile << std::fixed;

  try
  {
    parameterFile.open(parameterFileName.c_str(), std::ofstream::out);
  }
  catch (const std::ios_base::failure & e)
  {
    itkGenericExceptionMacro("Error opening parameter file: " << e.what());
  }

  try
  {
    ParameterMapConstIterator parameterMapIterator = parameterMap.begin();
    ParameterMapConstIterator parameterMapIteratorEnd = parameterMap.end();
    while (parameterMapIterator != parameterMapIteratorEnd)
    {
      parameterFile << "(" << parameterMapIterator->first;

      ParameterValueVectorType parameterMapValueVector = parameterMapIterator->second;
      for (unsigned int i = 0; i < parameterMapValueVector.size(); ++i)
      {
        std::istringstream stream(parameterMapValueVector[i]);
        float              number;
        stream >> number;
        if (stream.fail() || stream.bad())
        {
          parameterFile << " \"" << parameterMapValueVector[i] << "\"";
        }
        else
        {
          parameterFile << " " << number;
        }
      }

      parameterFile << ")" << std::endl;
      ++parameterMapIterator;
    }
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


/**
 * ********************* WriteParameterFile *********************
 */

void
ParameterObject::WriteParameterFile(const ParameterFileNameType & parameterFileName) const
{
  if (m_ParameterMaps.empty())
  {
    itkExceptionMacro("Error writing parameter map to disk: The parameter object is empty.");
  }

  if (m_ParameterMaps.size() > 1)
  {
    itkExceptionMacro(
      << "Error writing to disk: The number of parameter maps (" << m_ParameterMaps.size()
      << ") does not match the number of provided filenames (1). Please provide a vector of filenames.");
  }

  this->WriteParameterFile(m_ParameterMaps[0], parameterFileName);
}


/**
 * ********************* WriteParameterFile *********************
 */

void
ParameterObject::WriteParameterFile(const ParameterMapVectorType &      parameterMapVector,
                                    const ParameterFileNameVectorType & parameterFileNameVector)
{
  if (parameterMapVector.size() != parameterFileNameVector.size())
  {
    itkGenericExceptionMacro(<< "Error writing to disk: The number of parameter maps (" << parameterMapVector.size()
                             << ") does not match the number of provided filenames (" << parameterFileNameVector.size()
                             << ").");
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


/**
 * ********************* WriteParameterFile *********************
 */


void
ParameterObject::WriteParameterFile(const ParameterFileNameVectorType & parameterFileNameVector) const
{
  Self::WriteParameterFile(m_ParameterMaps, parameterFileNameVector);
}


/**
 * ********************* GetDefaultParameterMap *********************
 */

const ParameterObject::ParameterMapType
ParameterObject::GetDefaultParameterMap(const std::string & transformName,
                                        const unsigned int  numberOfResolutions,
                                        const double        finalGridSpacingInPhysicalUnits)
{
  // Parameters that depend on size and number of resolutions
  ParameterMapType parameterMap = ParameterMapType();

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
    ParameterValueVectorType gridSpacingSchedule = ParameterValueVectorType();
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
    ParameterMapConstIterator parameterMapIterator = m_ParameterMaps[i].begin();
    ParameterMapConstIterator parameterMapIteratorEnd = m_ParameterMaps[i].end();
    while (parameterMapIterator != parameterMapIteratorEnd)
    {
      os << "  (" << parameterMapIterator->first;
      ParameterValueVectorType parameterMapValueVector = parameterMapIterator->second;

      for (unsigned int j = 0; j < parameterMapValueVector.size(); ++j)
      {
        std::istringstream stream(parameterMapValueVector[j]);
        float              number;
        stream >> number;
        if (stream.fail())
        {
          os << " \"" << parameterMapValueVector[j] << "\"";
        }
        else
        {
          os << " " << number;
        }
      }

      os << ")" << std::endl;
      ++parameterMapIterator;
    }
  }
}


} // namespace elastix

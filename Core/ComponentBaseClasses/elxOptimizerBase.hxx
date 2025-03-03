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
#ifndef elxOptimizerBase_hxx
#define elxOptimizerBase_hxx

#include "elxOptimizerBase.h"
#include <itkDeref.h>

#include "itkSingleValuedNonLinearOptimizer.h"
#include "itk_zlib.h"
#include <cmath> // For round.

namespace elastix
{

/**
 * ****************** SetCurrentPositionPublic ************************
 */

template <typename TElastix>
void
OptimizerBase<TElastix>::SetCurrentPositionPublic(const ParametersType & /** param */)
{
  log::error(std::ostringstream{}
             << "ERROR: This function should be overridden or just not used.\n"
             << "  Are you using BSplineTransformWithDiffusion in combination with another optimizer than the "
                "StandardGradientDescentOptimizer? Don't!");

  /** Throw an exception if this function is not overridden. */
  itkExceptionMacro("ERROR: The SetCurrentPositionPublic method is not implemented in your optimizer");

} // end SetCurrentPositionPublic()


/**
 * ****************** BeforeEachResolutionBase **********************
 */

template <typename TElastix>
void
OptimizerBase<TElastix>::BeforeEachResolutionBase()
{
  /** Get the current resolution level. */
  unsigned int level = this->GetRegistration()->GetAsITKBaseType()->GetCurrentLevel();

  const Configuration & configuration = itk::Deref(Superclass::GetConfiguration());

  /** Check if after every iteration a new sample set should be created. */
  this->m_NewSamplesEveryIteration = false;
  configuration.ReadParameter(
    this->m_NewSamplesEveryIteration, "NewSamplesEveryIteration", this->GetComponentLabel(), level, 0);

} // end BeforeEachResolutionBase()


/**
 * ****************** AfterRegistrationBase **********************
 */

template <typename TElastix>
void
OptimizerBase<TElastix>::AfterRegistrationBase()
{
  using ParametersValueType = typename ParametersType::ValueType;

  /** Get the final parameters, round to six decimals. */
  ParametersType      finalTP = this->GetAsITKBaseType()->GetCurrentPosition();
  const unsigned long N = finalTP.GetSize();
  ParametersType      roundedTP(N);
  for (unsigned int i = 0; i < N; ++i)
  {
    roundedTP[i] = std::round(finalTP[i] * 1.0e6);
  }

  /** Compute the crc checksum using zlib crc32 function. */
  const auto * crcInputData = reinterpret_cast<const unsigned char *>(roundedTP.data_block());
  uLong        crc = crc32(0L, Z_NULL, 0);
  crc = crc32(crc, crcInputData, N * sizeof(ParametersValueType));

  log::info(std::ostringstream{} << "\nRegistration result checksum: " << crc);

} // end AfterRegistrationBase()


/**
 * ****************** SelectNewSamples ****************************
 */

template <typename TElastix>
void
OptimizerBase<TElastix>::SelectNewSamples()
{
  /** Force the metric to base its computation on a new subset of image samples.
   * Not every metric may have implemented this.
   */
  for (unsigned int i = 0; i < this->GetElastix()->GetNumberOfMetrics(); ++i)
  {
    this->GetElastix()->GetElxMetricBase(i)->SelectNewSamples();
  }

} // end SelectNewSamples()


/**
 * ****************** GetNewSamplesEveryIteration ********************
 */

template <typename TElastix>
bool
OptimizerBase<TElastix>::GetNewSamplesEveryIteration() const
{
  /** itkGetConstMacro Without the itkDebugMacro. */
  return this->m_NewSamplesEveryIteration;

} // end GetNewSamplesEveryIteration()


/**
 * **************** PrintSettingsVector **********************
 */

template <typename TElastix>
void
OptimizerBase<TElastix>::PrintSettingsVector(const SettingsVectorType & settings)
{
  const unsigned long nrofres = settings.size();

  std::ostringstream outputStringStream;
  outputStringStream << std::showpoint << std::fixed;

  /** Print to log file */
  outputStringStream << "( SP_a ";
  for (unsigned int i = 0; i < nrofres; ++i)
  {
    outputStringStream << settings[i].a << " ";
  }
  outputStringStream << ")\n"

                     << "( SP_A ";
  for (unsigned int i = 0; i < nrofres; ++i)
  {
    outputStringStream << settings[i].A << " ";
  }
  outputStringStream << ")\n"

                     << "( SP_alpha ";
  for (unsigned int i = 0; i < nrofres; ++i)
  {
    outputStringStream << settings[i].alpha << " ";
  }
  outputStringStream << ")\n"

                     << "( SigmoidMax ";
  for (unsigned int i = 0; i < nrofres; ++i)
  {
    outputStringStream << settings[i].fmax << " ";
  }
  outputStringStream << ")\n"

                     << "( SigmoidMin ";
  for (unsigned int i = 0; i < nrofres; ++i)
  {
    outputStringStream << settings[i].fmin << " ";
  }
  outputStringStream << ")\n"

                     << "( SigmoidScale ";
  for (unsigned int i = 0; i < nrofres; ++i)
  {
    outputStringStream << settings[i].omega << " ";
  }
  outputStringStream << ")\n";

  log::info(outputStringStream.str());

} // end PrintSettingsVector()


/**
 * ****************** SetSinusScales ********************
 */

template <typename TElastix>
void
OptimizerBase<TElastix>::SetSinusScales(double amplitude, double frequency, unsigned long numberOfParameters)
{
  using ScalesType = typename ITKBaseType::ScalesType;

  const auto nrofpar = static_cast<double>(numberOfParameters);
  ScalesType scales(numberOfParameters);

  for (unsigned long i = 0; i < numberOfParameters; ++i)
  {
    const double x = static_cast<double>(i) / nrofpar * 2.0 * vnl_math::pi * frequency;
    scales[i] = std::pow(amplitude, std::sin(x));
  }
  this->GetAsITKBaseType()->SetScales(scales);

} // end SetSinusScales()


} // end namespace elastix

#endif // end #ifndef elxOptimizerBase_hxx

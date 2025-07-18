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
#ifndef itkImageRandomSamplerBase_hxx
#define itkImageRandomSamplerBase_hxx

#include "itkImageRandomSamplerBase.h"

#include "itkMersenneTwisterRandomVariateGenerator.h"

namespace itk
{

/**
 * ******************* Constructor *******************
 */

template <typename TInputImage>
ImageRandomSamplerBase<TInputImage>::ImageRandomSamplerBase()
{
  this->m_NumberOfSamples = 1000;

} // end Constructor


/**
 * ******************* GenerateRandomNumberList *******************
 */

template <typename TInputImage>
void
ImageRandomSamplerBase<TInputImage>::GenerateRandomNumberList()
{
  elx::DefaultConstruct<Statistics::MersenneTwisterRandomVariateGenerator> randomVariateGenerator{};
  randomVariateGenerator.SetSeed(m_OptionalSeed.value_or(++m_Seed));

  /** Clear the random number list. */
  this->m_RandomNumberList.clear();
  this->m_RandomNumberList.reserve(this->m_NumberOfSamples);

  /** Fill the list with random numbers. */
  const auto numPixels = static_cast<double>(this->GetCroppedInputImageRegion().GetNumberOfPixels());
  randomVariateGenerator.GetVariateWithOpenRange(numPixels - 0.5); // dummy jump
  for (unsigned long i = 0; i < this->m_NumberOfSamples; ++i)
  {
    const double randomPosition = randomVariateGenerator.GetVariateWithOpenRange(numPixels - 0.5);
    this->m_RandomNumberList.push_back(randomPosition);
  }
}

/**
 * ******************* PrintSelf *******************
 */

template <typename TInputImage>
void
ImageRandomSamplerBase<TInputImage>::PrintSelf(std::ostream & os, Indent indent) const
{
  Superclass::PrintSelf(os, indent);

  os << indent << "NumberOfSamples: " << this->m_NumberOfSamples << std::endl;

} // end PrintSelf()


} // end namespace itk

#endif // end #ifndef itkImageRandomSamplerBase_hxx

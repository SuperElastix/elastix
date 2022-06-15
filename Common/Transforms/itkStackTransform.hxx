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
#ifndef _itkStackTransform_hxx
#define _itkStackTransform_hxx

#include "itkStackTransform.h"

namespace itk
{

/**
 * ************************ SetParameters ***********************
 */

template <class TScalarType, unsigned int NInputDimensions, unsigned int NOutputDimensions>
void
StackTransform<TScalarType, NInputDimensions, NOutputDimensions>::SetParameters(const ParametersType & param)
{
  // All subtransforms should be the same and should have the same number of parameters.
  // Here we check if the number of parameters is #subtransforms * #parameters per subtransform.
  if (param.GetSize() != this->GetNumberOfParameters())
  {
    itkExceptionMacro(<< "Number of parameters does not match the number of subtransforms * the number of parameters "
                         "per subtransform.");
  }

  // Set separate subtransform parameters
  const NumberOfParametersType numSubTransformParameters = this->m_SubTransformContainer[0]->GetNumberOfParameters();
  const auto                   numberOfSubTransforms = static_cast<unsigned>(m_SubTransformContainer.size());
  for (unsigned int t = 0; t < numberOfSubTransforms; ++t)
  {
    // NTA, split the parameter by number of subparameters
    const ParametersType subparams(&(param.data_block()[t * numSubTransformParameters]), numSubTransformParameters);
    this->m_SubTransformContainer[t]->SetParametersByValue(subparams);
  }

  this->Modified();
} // end SetParameters()


/**
 * ************************ GetParameters ***********************
 */

template <class TScalarType, unsigned int NInputDimensions, unsigned int NOutputDimensions>
auto
StackTransform<TScalarType, NInputDimensions, NOutputDimensions>::GetParameters() const -> const ParametersType &
{
  this->m_Parameters.SetSize(this->GetNumberOfParameters());

  // Fill params with parameters of subtransforms
  unsigned int i = 0;
  for (const auto & subTransform : m_SubTransformContainer)
  {
    const auto numberOfSubTransformParameters = this->m_SubTransformContainer[0]->GetNumberOfParameters();

    const ParametersType & subparams = subTransform->GetParameters();
    for (unsigned int p = 0; p < numberOfSubTransformParameters; ++p, ++i)
    {
      this->m_Parameters[i] = subparams[p];
    }
  }

  return this->m_Parameters;
} // end GetParameters()


/**
 * ********************* TransformPoint ****************************
 */

template <class TScalarType, unsigned int NInputDimensions, unsigned int NOutputDimensions>
auto
StackTransform<TScalarType, NInputDimensions, NOutputDimensions>::TransformPoint(
  const InputPointType & inputPoint) const -> OutputPointType
{
  /** Reduce dimension of input point. */
  SubTransformInputPointType ippr;
  for (unsigned int d = 0; d < ReducedInputSpaceDimension; ++d)
  {
    ippr[d] = inputPoint[d];
  }

  /** Transform point using right subtransform. */
  SubTransformOutputPointType oppr;
  const unsigned int          subt =
    std::min(static_cast<unsigned int>(this->m_SubTransformContainer.size() - 1),
             static_cast<unsigned int>(
               std::max(0, vnl_math::rnd((inputPoint[ReducedInputSpaceDimension] - m_StackOrigin) / m_StackSpacing))));
  oppr = this->m_SubTransformContainer[subt]->TransformPoint(ippr);

  /** Increase dimension of input point. */
  OutputPointType opp;
  for (unsigned int d = 0; d < ReducedOutputSpaceDimension; ++d)
  {
    opp[d] = oppr[d];
  }
  opp[ReducedOutputSpaceDimension] = inputPoint[ReducedInputSpaceDimension];

  return opp;

} // end TransformPoint()


/**
 * ********************* GetJacobian ****************************
 */

template <class TScalarType, unsigned int NInputDimensions, unsigned int NOutputDimensions>
void
StackTransform<TScalarType, NInputDimensions, NOutputDimensions>::GetJacobian(const InputPointType &       inputPoint,
                                                                              JacobianType &               jac,
                                                                              NonZeroJacobianIndicesType & nzji) const
{
  /** Reduce dimension of input point. */
  SubTransformInputPointType ippr;
  for (unsigned int d = 0; d < ReducedInputSpaceDimension; ++d)
  {
    ippr[d] = inputPoint[d];
  }

  /** Get Jacobian from right subtransform. */
  const unsigned int subt =
    std::min(static_cast<unsigned int>(this->m_SubTransformContainer.size() - 1),
             static_cast<unsigned int>(
               std::max(0, vnl_math::rnd((inputPoint[ReducedInputSpaceDimension] - m_StackOrigin) / m_StackSpacing))));
  SubTransformJacobianType subjac;
  this->m_SubTransformContainer[subt]->GetJacobian(ippr, subjac, nzji);

  /** Fill output Jacobian. */
  jac.set_size(InputSpaceDimension, nzji.size());
  jac.Fill(0.0);
  for (unsigned int d = 0; d < ReducedInputSpaceDimension; ++d)
  {
    for (unsigned int n = 0; n < nzji.size(); ++n)
    {
      jac[d][n] = subjac[d][n];
    }
  }

  /** Update non zero Jacobian indices. */
  for (unsigned int i = 0; i < nzji.size(); ++i)
  {
    nzji[i] += subt * this->m_SubTransformContainer[0]->GetNumberOfParameters();
  }

} // end GetJacobian()


/**
 * ********************* GetNumberOfNonZeroJacobianIndices ****************************
 */

template <class TScalarType, unsigned int NInputDimensions, unsigned int NOutputDimensions>
auto
StackTransform<TScalarType, NInputDimensions, NOutputDimensions>::GetNumberOfNonZeroJacobianIndices() const
  -> NumberOfParametersType
{
  return this->m_SubTransformContainer[0]->GetNumberOfNonZeroJacobianIndices();

} // end GetNumberOfNonZeroJacobianIndices()


} // end namespace itk

#endif

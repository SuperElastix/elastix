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
/*=========================================================================

  Program:   Insight Segmentation & Registration Toolkit
  Module:    $RCSfile: itkThinPlateR2LogRSplineKernelTransform2.txx,v $
  Language:  C++
  Date:      $Date: 2006/03/19 04:36:59 $
  Version:   $Revision: 1.8 $

  Copyright (c) Insight Software Consortium. All rights reserved.
  See ITKCopyright.txt or http://www.itk.org/HTML/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notices for more information.

=========================================================================*/
#ifndef _itkThinPlateR2LogRSplineKernelTransform2_hxx
#define _itkThinPlateR2LogRSplineKernelTransform2_hxx

#include "itkThinPlateR2LogRSplineKernelTransform2.h"

namespace itk
{

template <class TScalarType, unsigned int NDimensions>
void
ThinPlateR2LogRSplineKernelTransform2<TScalarType, NDimensions>::ComputeG(const InputVectorType & x,
                                                                          GMatrixType &           GMatrix) const
{
  const TScalarType r = x.GetNorm();
  GMatrix.fill(NumericTraits<TScalarType>::ZeroValue());
  const TScalarType R2logR = (r > 1e-8) ? r * r * std::log(r) : NumericTraits<TScalarType>::Zero;

  GMatrix.fill_diagonal(R2logR);
}


template <class TScalarType, unsigned int NDimensions>
void
ThinPlateR2LogRSplineKernelTransform2<TScalarType, NDimensions>::ComputeDeformationContribution(
  const InputPointType & thisPoint,
  OutputPointType &      result) const
{
  const unsigned long numberOfLandmarks = this->m_SourceLandmarks->GetNumberOfPoints();

  PointsIterator sp = this->m_SourceLandmarks->GetPoints()->Begin();

  for (unsigned int lnd = 0; lnd < numberOfLandmarks; ++lnd)
  {
    InputVectorType   position = thisPoint - sp->Value();
    const TScalarType r = position.GetNorm();
    const TScalarType R2logR = (r > 1e-8) ? r * r * std::log(r) : NumericTraits<TScalarType>::Zero;
    for (unsigned int odim = 0; odim < NDimensions; ++odim)
    {
      result[odim] += R2logR * this->m_DMatrix(odim, lnd);
    }
    ++sp;
  }
}


} // namespace itk

#endif

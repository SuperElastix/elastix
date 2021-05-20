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
  Module:    $RCSfile: itkThinPlateSplineKernelTransform2.txx,v $
  Language:  C++
  Date:      $Date: 2006-11-28 14:22:18 $
  Version:   $Revision: 1.1 $

  Copyright (c) Insight Software Consortium. All rights reserved.
  See ITKCopyright.txt or http://www.itk.org/HTML/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notices for more information.

=========================================================================*/
#ifndef _itkThinPlateSplineKernelTransform2_hxx
#define _itkThinPlateSplineKernelTransform2_hxx

#include "itkThinPlateSplineKernelTransform2.h"

namespace itk
{

/**
 * ******************* Constructor *******************
 */

template <class TScalarType, unsigned int NDimensions>
void
ThinPlateSplineKernelTransform2<TScalarType, NDimensions>::ComputeG(const InputVectorType & x,
                                                                    GMatrixType &           GMatrix) const
{
  const TScalarType r = x.GetNorm();
  GMatrix.fill(NumericTraits<TScalarType>::ZeroValue());
  GMatrix.fill_diagonal(r);

} // end ComputeG()


/**
 * ******************* ComputeDeformationContribution *******************
 */

template <class TScalarType, unsigned int NDimensions>
void
ThinPlateSplineKernelTransform2<TScalarType, NDimensions>::ComputeDeformationContribution(
  const InputPointType & thisPoint,
  OutputPointType &      opp) const
{
  const unsigned long numberOfLandmarks = this->m_SourceLandmarks->GetNumberOfPoints();
  PointsIterator      sp = this->m_SourceLandmarks->GetPoints()->Begin();

  for (unsigned long lnd = 0; lnd < numberOfLandmarks; ++lnd)
  {
    InputVectorType   position = thisPoint - sp->Value();
    const TScalarType r = position.GetNorm();

    for (unsigned int odim = 0; odim < NDimensions; ++odim)
    {
      opp[odim] += r * this->m_DMatrix(odim, lnd);
    }
    ++sp;
  }

} // end ComputeDeformationContribution()


} // namespace itk

#endif

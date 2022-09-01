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
  Module:    $RCSfile: itkSingleValuedPointSetToPointSetMetric.txx,v $
  Language:  C++
  Date:      $Date: 2009-01-26 21:45:56 $
  Version:   $Revision: 1.2 $

  Copyright (c) Insight Software Consortium. All rights reserved.
  See ITKCopyright.txt or http://www.itk.org/HTML/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notices for more information.

=========================================================================*/
#ifndef itkSingleValuedPointSetToPointSetMetric_hxx
#define itkSingleValuedPointSetToPointSetMetric_hxx

#include "itkSingleValuedPointSetToPointSetMetric.h"

namespace itk
{

/**
 * ******************* SetTransformParameters ***********************
 */

template <class TFixedPointSet, class TMovingPointSet>
void
SingleValuedPointSetToPointSetMetric<TFixedPointSet, TMovingPointSet>::SetTransformParameters(
  const ParametersType & parameters) const
{
  if (!this->m_Transform)
  {
    itkExceptionMacro(<< "Transform has not been assigned");
  }
  this->m_Transform->SetParameters(parameters);

} // end SetTransformParameters()


/**
 * ******************* Initialize ***********************
 */

template <class TFixedPointSet, class TMovingPointSet>
void
SingleValuedPointSetToPointSetMetric<TFixedPointSet, TMovingPointSet>::Initialize()
{
  if (!this->m_Transform)
  {
    itkExceptionMacro(<< "Transform is not present");
  }

  if (!this->m_MovingPointSet)
  {
    itkExceptionMacro(<< "MovingPointSet is not present");
  }

  if (!this->m_FixedPointSet)
  {
    itkExceptionMacro(<< "FixedPointSet is not present");
  }

  // If the PointSet is provided by a source, update the source.
  if (this->m_MovingPointSet->GetSource())
  {
    this->m_MovingPointSet->GetSource()->Update();
  }

  // If the point set is provided by a source, update the source.
  if (this->m_FixedPointSet->GetSource())
  {
    this->m_FixedPointSet->GetSource()->Update();
  }

} // end Initialize()


/**
 * *********************** BeforeThreadedGetValueAndDerivative ***********************
 */

template <class TFixedPointSet, class TMovingPointSet>
void
SingleValuedPointSetToPointSetMetric<TFixedPointSet, TMovingPointSet>::BeforeThreadedGetValueAndDerivative(
  const TransformParametersType & parameters) const
{
  /** In this function do all stuff that cannot be multi-threaded. */
  if (this->m_UseMetricSingleThreaded)
  {
    this->SetTransformParameters(parameters);
  }

} // end BeforeThreadedGetValueAndDerivative()


/**
 * ******************* PrintSelf ***********************
 */

template <class TFixedPointSet, class TMovingPointSet>
void
SingleValuedPointSetToPointSetMetric<TFixedPointSet, TMovingPointSet>::PrintSelf(std::ostream & os, Indent indent) const
{
  Superclass::PrintSelf(os, indent);

  os << "Fixed  PointSet: " << this->m_FixedPointSet.GetPointer() << std::endl;
  os << "Moving PointSet: " << this->m_MovingPointSet.GetPointer() << std::endl;
  os << "Fixed mask: " << this->m_FixedImageMask.GetPointer() << std::endl;
  os << "Moving mask: " << this->m_MovingImageMask.GetPointer() << std::endl;
  os << "Transform: " << this->m_Transform.GetPointer() << std::endl;

} // end PrintSelf()


} // end namespace itk

#endif // end #ifndef itkSingleValuedPointSetToPointSetMetric_hxx

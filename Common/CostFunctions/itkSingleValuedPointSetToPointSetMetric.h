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
  Module:    $RCSfile: itkSingleValuedPointSetToPointSetMetric.h,v $
  Language:  C++
  Date:      $Date: 2009-01-26 21:45:56 $
  Version:   $Revision: 1.2 $

  Copyright (c) Insight Software Consortium. All rights reserved.
  See ITKCopyright.txt or http://www.itk.org/HTML/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notices for more information.

=========================================================================*/
#ifndef itkSingleValuedPointSetToPointSetMetric_h
#define itkSingleValuedPointSetToPointSetMetric_h

#include "itkImageBase.h"
#include "itkAdvancedTransform.h"
#include "itkSingleValuedCostFunction.h"
#include "itkMacro.h"
#include "itkSpatialObject.h"
#include "itkPointSet.h"

namespace itk
{

/** \class SingleValuedPointSetToPointSetMetric
 * \brief Computes similarity between two point sets.
 *
 * This Class is templated over the type of the two point-sets. It
 * expects a Transform to be plugged in. This particular
 * class is the base class for a hierarchy of point-set to point-set metrics.
 *
 * This class computes a value that measures the similarity between the fixed point-set
 * and the transformed moving point-set.
 *
 * \ingroup RegistrationMetrics
 *
 */

template <class TFixedPointSet, class TMovingPointSet>
class ITK_TEMPLATE_EXPORT SingleValuedPointSetToPointSetMetric : public SingleValuedCostFunction
{
public:
  ITK_DISALLOW_COPY_AND_MOVE(SingleValuedPointSetToPointSetMetric);

  /** Standard class typedefs. */
  using Self = SingleValuedPointSetToPointSetMetric;
  using Superclass = SingleValuedCostFunction;
  using Pointer = SmartPointer<Self>;
  using ConstPointer = SmartPointer<const Self>;

  /** Type used for representing point components  */
  using CoordinateRepresentationType = Superclass::ParametersValueType;

  /** Run-time type information (and related methods). */
  itkTypeMacro(SingleValuedPointSetToPointSetMetric, SingleValuedCostFunction);

  /** Typedefs. */
  using FixedPointSetType = TFixedPointSet;
  using FixedPointSetPixelType = typename FixedPointSetType::PixelType;
  using FixedPointSetConstPointer = typename FixedPointSetType::ConstPointer;
  using MovingPointSetType = TMovingPointSet;
  using MovingPointSetPixelType = typename MovingPointSetType::PixelType;
  using MovingPointSetConstPointer = typename MovingPointSetType::ConstPointer;
  using PointIterator = typename FixedPointSetType::PointsContainer::ConstIterator;
  using PointDataIterator = typename FixedPointSetType::PointDataContainer::ConstIterator;

  /** Constants for the pointset dimensions. */
  itkStaticConstMacro(FixedPointSetDimension, unsigned int, TFixedPointSet::PointDimension);
  itkStaticConstMacro(MovingPointSetDimension, unsigned int, TMovingPointSet::PointDimension);

  /**  More typedefs. */
  using TransformType =
    AdvancedTransform<CoordinateRepresentationType, Self::FixedPointSetDimension, Self::MovingPointSetDimension>;
  using TransformPointer = typename TransformType::Pointer;
  using InputPointType = typename TransformType::InputPointType;
  using OutputPointType = typename TransformType::OutputPointType;
  using TransformParametersType = typename TransformType::ParametersType;
  using TransformJacobianType = typename TransformType::JacobianType;

  using FixedImageMaskType = SpatialObject<Self::FixedPointSetDimension>;
  using FixedImageMaskPointer = typename FixedImageMaskType::Pointer;
  using FixedImageMaskConstPointer = typename FixedImageMaskType::ConstPointer;
  using MovingImageMaskType = SpatialObject<Self::MovingPointSetDimension>;
  using MovingImageMaskPointer = typename MovingImageMaskType::Pointer;
  using MovingImageMaskConstPointer = typename MovingImageMaskType::ConstPointer;

  /**  Type of the measure. */
  using Superclass::MeasureType;
  using Superclass::DerivativeType;
  using DerivativeValueType = typename DerivativeType::ValueType;
  using Superclass::ParametersType;

  /** Typedefs for support of sparse Jacobians and compact support of transformations. */
  using NonZeroJacobianIndicesType = typename TransformType::NonZeroJacobianIndicesType;

  /** Connect the fixed pointset.  */
  itkSetConstObjectMacro(FixedPointSet, FixedPointSetType);

  /** Get the fixed pointset. */
  itkGetConstObjectMacro(FixedPointSet, FixedPointSetType);

  /** Connect the moving pointset.  */
  itkSetConstObjectMacro(MovingPointSet, MovingPointSetType);

  /** Get the moving pointset. */
  itkGetConstObjectMacro(MovingPointSet, MovingPointSetType);

  /** Connect the Transform. */
  itkSetObjectMacro(Transform, TransformType);

  /** Get a pointer to the Transform.  */
  itkGetConstObjectMacro(Transform, TransformType);

  /** Set the parameters defining the Transform. */
  void
  SetTransformParameters(const ParametersType & parameters) const;

  /** Return the number of parameters required by the transform. */
  unsigned int
  GetNumberOfParameters() const override
  {
    return this->m_Transform->GetNumberOfParameters();
  }

  /** Initialize the Metric by making sure that all the components are
   *  present and plugged together correctly.
   */
  virtual void
  Initialize();

  /** Set the fixed mask. */
  // \todo: currently not used
  itkSetConstObjectMacro(FixedImageMask, FixedImageMaskType);

  /** Get the fixed mask. */
  itkGetConstObjectMacro(FixedImageMask, FixedImageMaskType);

  /** Set the moving mask. */
  itkSetConstObjectMacro(MovingImageMask, MovingImageMaskType);

  /** Get the moving mask. */
  itkGetConstObjectMacro(MovingImageMask, MovingImageMaskType);

  /** Contains calls from GetValueAndDerivative that are thread-unsafe. */
  virtual void
  BeforeThreadedGetValueAndDerivative(const TransformParametersType & parameters) const;

  /** Switch the function BeforeThreadedGetValueAndDerivative on or off. */
  itkSetMacro(UseMetricSingleThreaded, bool);
  itkGetConstReferenceMacro(UseMetricSingleThreaded, bool);
  itkBooleanMacro(UseMetricSingleThreaded);

protected:
  SingleValuedPointSetToPointSetMetric() = default;
  ~SingleValuedPointSetToPointSetMetric() override = default;

  /** PrintSelf. */
  void
  PrintSelf(std::ostream & os, Indent indent) const override;

  /** Member variables. */
  FixedPointSetConstPointer   m_FixedPointSet{ nullptr };
  MovingPointSetConstPointer  m_MovingPointSet{ nullptr };
  FixedImageMaskConstPointer  m_FixedImageMask{ nullptr };
  MovingImageMaskConstPointer m_MovingImageMask{ nullptr };
  mutable TransformPointer    m_Transform{ nullptr };

  mutable unsigned int m_NumberOfPointsCounted{ 0 };

  /** Variables for multi-threading. */
  bool m_UseMetricSingleThreaded{ true };
};

} // end namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#  include "itkSingleValuedPointSetToPointSetMetric.hxx"
#endif

#endif

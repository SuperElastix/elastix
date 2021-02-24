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
#ifndef itkDeformationFieldInterpolatingTransform_h
#define itkDeformationFieldInterpolatingTransform_h

#include <iostream>
#include "itkAdvancedTransform.h"
#include "itkMacro.h"
#include "itkImage.h"
#include "itkVectorInterpolateImageFunction.h"
#include "itkVectorNearestNeighborInterpolateImageFunction.h"

namespace itk
{

/** \brief Transform that interpolates a given deformation field
 *
 * A simple transform that allows the user to set a deformation field.
 * TransformPoint adds the displacement to the input point.
 * This transform does not support optimizers. Its Set/GetParameters
 * is not implemented. DO NOT USE IT FOR REGISTRATION.
 * You may set your own interpolator!
 *
 * \ingroup Transforms
 */

template <class TScalarType = double,   // Data type for scalars (float or double)
          unsigned int NDimensions = 3, // Number of input dimensions
          class TComponentType = double>
// ComponentType of the deformation field
class ITK_TEMPLATE_EXPORT DeformationFieldInterpolatingTransform
  : public AdvancedTransform<TScalarType, NDimensions, NDimensions>
{
public:
  /** Standard class typedefs. */
  typedef DeformationFieldInterpolatingTransform                   Self;
  typedef AdvancedTransform<TScalarType, NDimensions, NDimensions> Superclass;
  typedef SmartPointer<Self>                                       Pointer;
  typedef SmartPointer<const Self>                                 ConstPointer;

  /** New macro for creation of through the object factory.*/
  itkNewMacro(Self);

  /** Run-time type information (and related methods). */
  itkTypeMacro(DeformationFieldInterpolatingTransform, AdvancedTransform);

  /** Dimension of the domain spaces. */
  itkStaticConstMacro(InputSpaceDimension, unsigned int, Superclass::InputSpaceDimension);
  itkStaticConstMacro(OutputSpaceDimension, unsigned int, Superclass::OutputSpaceDimension);

  /** Superclass typedefs */
  typedef typename Superclass::ScalarType                    ScalarType;
  typedef typename Superclass::ParametersType                ParametersType;
  typedef typename Superclass::NumberOfParametersType        NumberOfParametersType;
  typedef typename Superclass::JacobianType                  JacobianType;
  typedef typename Superclass::InputVectorType               InputVectorType;
  typedef typename Superclass::OutputVectorType              OutputVectorType;
  typedef typename Superclass::InputCovariantVectorType      InputCovariantVectorType;
  typedef typename Superclass::OutputCovariantVectorType     OutputCovariantVectorType;
  typedef typename Superclass::InputVnlVectorType            InputVnlVectorType;
  typedef typename Superclass::OutputVnlVectorType           OutputVnlVectorType;
  typedef typename Superclass::InputPointType                InputPointType;
  typedef typename Superclass::OutputPointType               OutputPointType;
  typedef typename Superclass::NonZeroJacobianIndicesType    NonZeroJacobianIndicesType;
  typedef typename Superclass::SpatialHessianType            SpatialHessianType;
  typedef typename Superclass::SpatialJacobianType           SpatialJacobianType;
  typedef typename Superclass::JacobianOfSpatialHessianType  JacobianOfSpatialHessianType;
  typedef typename Superclass::JacobianOfSpatialJacobianType JacobianOfSpatialJacobianType;

  typedef typename Superclass::InternalMatrixType InternalMatrixType;

  typedef TComponentType DeformationFieldComponentType;
  typedef Vector<DeformationFieldComponentType, itkGetStaticConstMacro(OutputSpaceDimension)>
                                                                                         DeformationFieldVectorType;
  typedef Image<DeformationFieldVectorType, itkGetStaticConstMacro(InputSpaceDimension)> DeformationFieldType;
  typedef typename DeformationFieldType::Pointer                                         DeformationFieldPointer;

  typedef VectorInterpolateImageFunction<DeformationFieldType, ScalarType> DeformationFieldInterpolatorType;
  typedef typename DeformationFieldInterpolatorType::Pointer               DeformationFieldInterpolatorPointer;
  typedef VectorNearestNeighborInterpolateImageFunction<DeformationFieldType, ScalarType>
    DefaultDeformationFieldInterpolatorType;

  /** Set the transformation parameters is not supported.
   * Use SetDeformationField() instead
   */
  void
  SetParameters(const ParametersType &) override
  {
    itkExceptionMacro(<< "ERROR: SetParameters() is not implemented "
                      << "for DeformationFieldInterpolatingTransform.\n"
                      << "Use SetDeformationField() instead.\n"
                      << "Note that this transform is NOT suited for image registration.\n"
                      << "Just use it as an (initial) fixed transform that is not optimized.");
  }


  /** Set the fixed parameters. */
  void
  SetFixedParameters(const ParametersType &) override
  {
    // This transform has no fixed parameters.
  }


  /** Get the Fixed Parameters. */
  const ParametersType &
  GetFixedParameters(void) const override
  {
    // This transform has no fixed parameters.
    return this->m_FixedParameters;
  }


  /** Transform a point. This method adds a displacement to a given point,
   * returning the transformed point.
   */
  OutputPointType
  TransformPoint(const InputPointType & point) const override;

  /** These vector transforms are not implemented for this transform. */
  OutputVectorType
  TransformVector(const InputVectorType &) const override
  {
    itkExceptionMacro(<< "TransformVector(const InputVectorType &) is not implemented "
                      << "for DeformationFieldInterpolatingTransform");
  }


  OutputVnlVectorType
  TransformVector(const InputVnlVectorType &) const override
  {
    itkExceptionMacro(<< "TransformVector(const InputVnlVectorType &) is not implemented "
                      << "for DeformationFieldInterpolatingTransform");
  }


  OutputCovariantVectorType
  TransformCovariantVector(const InputCovariantVectorType &) const override
  {
    itkExceptionMacro(<< "TransformCovariantVector(const InputCovariantVectorType &) is not implemented "
                      << "for DeformationFieldInterpolatingTransform");
  }


  /** Make this an identity transform ( the deformation field is replaced
   * by a zero deformation field */
  void
  SetIdentity(void);

  /** Set/Get the deformation field that defines the displacements */
  virtual void
  SetDeformationField(DeformationFieldType * _arg);

  itkGetModifiableObjectMacro(DeformationField, DeformationFieldType);

  /** Set/Get the deformation field interpolator */
  virtual void
  SetDeformationFieldInterpolator(DeformationFieldInterpolatorType * _arg);

  itkGetModifiableObjectMacro(DeformationFieldInterpolator, DeformationFieldInterpolatorType);

  bool
  IsLinear(void) const override
  {
    return false;
  }

  /** Must be provided. */
  void
  GetJacobian(const InputPointType &       ipp,
              JacobianType &               j,
              NonZeroJacobianIndicesType & nonZeroJacobianIndices) const override
  {
    itkExceptionMacro(<< "Not implemented for DeformationFieldInterpolatingTransform");
  }


  void
  GetSpatialJacobian(const InputPointType & ipp, SpatialJacobianType & sj) const override
  {
    itkExceptionMacro(<< "Not implemented for DeformationFieldInterpolatingTransform");
  }


  void
  GetSpatialHessian(const InputPointType & ipp, SpatialHessianType & sh) const override
  {
    itkExceptionMacro(<< "Not implemented for DeformationFieldInterpolatingTransform");
  }


  void
  GetJacobianOfSpatialJacobian(const InputPointType &          ipp,
                               JacobianOfSpatialJacobianType & jsj,
                               NonZeroJacobianIndicesType &    nonZeroJacobianIndices) const override
  {
    itkExceptionMacro(<< "Not implemented for DeformationFieldInterpolatingTransform");
  }


  void
  GetJacobianOfSpatialJacobian(const InputPointType &          ipp,
                               SpatialJacobianType &           sj,
                               JacobianOfSpatialJacobianType & jsj,
                               NonZeroJacobianIndicesType &    nonZeroJacobianIndices) const override
  {
    itkExceptionMacro(<< "Not implemented for DeformationFieldInterpolatingTransform");
  }


  void
  GetJacobianOfSpatialHessian(const InputPointType &         ipp,
                              JacobianOfSpatialHessianType & jsh,
                              NonZeroJacobianIndicesType &   nonZeroJacobianIndices) const override
  {
    itkExceptionMacro(<< "Not implemented for DeformationFieldInterpolatingTransform");
  }


  void
  GetJacobianOfSpatialHessian(const InputPointType &         ipp,
                              SpatialHessianType &           sh,
                              JacobianOfSpatialHessianType & jsh,
                              NonZeroJacobianIndicesType &   nonZeroJacobianIndices) const override
  {
    itkExceptionMacro(<< "Not implemented for DeformationFieldInterpolatingTransform");
  }


protected:
  DeformationFieldInterpolatingTransform();
  ~DeformationFieldInterpolatingTransform() override;

  /** Typedef which is used internally */
  typedef typename DeformationFieldInterpolatorType::ContinuousIndexType InputContinuousIndexType;
  typedef typename DeformationFieldInterpolatorType::OutputType          InterpolatorOutputType;

  /** Print contents of an DeformationFieldInterpolatingTransform. */
  void
  PrintSelf(std::ostream & os, Indent indent) const override;

  DeformationFieldPointer             m_DeformationField;
  DeformationFieldPointer             m_ZeroDeformationField;
  DeformationFieldInterpolatorPointer m_DeformationFieldInterpolator;

private:
  DeformationFieldInterpolatingTransform(const Self &) = delete;
  void
  operator=(const Self &) = delete;
};

} // namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#  include "itkDeformationFieldInterpolatingTransform.hxx"
#endif

#endif /* itkDeformationFieldInterpolatingTransform_h */

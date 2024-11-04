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
#ifndef itkAdvancedTransformAdapter_h
#define itkAdvancedTransformAdapter_h

#include "itkAdvancedTransform.h"
#include "itkMacro.h"
#include <itkDeref.h>


namespace elastix
{

/** \brief Adapts the ITK transform that is specified by AdvancedTransformAdapter::SetExternalTransform to the elastix
 * AdvancedTransform interface.
 *
 * DO NOT USE IT FOR REGISTRATION. DO NOT USE IT TO RETRIEVE JACOBIAN OR THE HESSIAN VALUES.
 *
 * \ingroup Transforms
 */

template <typename TScalarType, unsigned int NDimensions>
class ITK_TEMPLATE_EXPORT AdvancedTransformAdapter
  : public itk::AdvancedTransform<TScalarType, NDimensions, NDimensions>
{
public:
  ITK_DISALLOW_COPY_AND_MOVE(AdvancedTransformAdapter);

  /** Standard class typedefs. */
  using Self = AdvancedTransformAdapter;
  using Superclass = itk::AdvancedTransform<TScalarType, NDimensions, NDimensions>;
  using Pointer = itk::SmartPointer<Self>;
  using ConstPointer = itk::SmartPointer<const Self>;

  /** New macro for creation of through the object factory.*/
  itkNewMacro(Self);

  /** Run-time type information (and related methods). */
  itkOverrideGetNameOfClassMacro(AdvancedTransformAdapter);

  /** Dimension of the domain spaces. */
  itkStaticConstMacro(InputSpaceDimension, unsigned int, Superclass::InputSpaceDimension);
  itkStaticConstMacro(OutputSpaceDimension, unsigned int, Superclass::OutputSpaceDimension);

  /** Superclass typedefs */
  using typename Superclass::ParametersType;
  using typename Superclass::JacobianType;
  using typename Superclass::InputVectorType;
  using typename Superclass::OutputVectorType;
  using typename Superclass::InputCovariantVectorType;
  using typename Superclass::OutputCovariantVectorType;
  using typename Superclass::InputVnlVectorType;
  using typename Superclass::OutputVnlVectorType;
  using typename Superclass::InputPointType;
  using typename Superclass::OutputPointType;
  using typename Superclass::NonZeroJacobianIndicesType;
  using typename Superclass::SpatialHessianType;
  using typename Superclass::SpatialJacobianType;
  using typename Superclass::JacobianOfSpatialHessianType;
  using typename Superclass::JacobianOfSpatialJacobianType;

  /** Set the transformation parameters. */
  void
  SetParameters(const ParametersType & parameters) override
  {
    // SetParameters may be called by elx::TransformBase<TElastix>::ReadFromFile(), but only for an empty list of
    // parameters _and_ an unspecified (null) m_ExternalTransform.
    if (m_ExternalTransform || !parameters.empty())
    {
      itkExceptionMacro("The parameters of an external transform cannot be set! Only the trivial case of setting an "
                        "empty parameters to an unspecified (null) external transform is supported!");
    }
  }

  /** Set the fixed parameters. Not implemented for this transform. */
  void
  SetFixedParameters(const ParametersType &) override
  {
    itkExceptionMacro(<< unimplementedOverrideMessage);
  }

  /** Get the fixed parameters. */
  const ParametersType &
  GetFixedParameters() const override
  {
    return itk::Deref(m_ExternalTransform.GetPointer()).GetFixedParameters();
  }

  /** Transform a point. */
  OutputPointType
  TransformPoint(const InputPointType & point) const override
  {
    return itk::Deref(m_ExternalTransform.GetPointer()).TransformPoint(point);
  }

  /** These vector transforms are not implemented for this transform. */
  OutputVectorType
  TransformVector(const InputVectorType &) const override
  {
    itkExceptionMacro(<< unimplementedOverrideMessage);
  }

  OutputVnlVectorType
  TransformVector(const InputVnlVectorType &) const override
  {
    itkExceptionMacro(<< unimplementedOverrideMessage);
  }

  OutputCovariantVectorType
  TransformCovariantVector(const InputCovariantVectorType &) const override
  {
    itkExceptionMacro(<< unimplementedOverrideMessage);
  }

  bool
  IsLinear() const override
  {
    return itk::Deref(m_ExternalTransform.GetPointer()).IsLinear();
  }

  /** Must be provided. */
  void
  GetJacobian(const InputPointType &, JacobianType &, NonZeroJacobianIndicesType &) const override
  {
    itkExceptionMacro(<< unimplementedOverrideMessage);
  }

  void
  GetSpatialJacobian(const InputPointType &, SpatialJacobianType &) const override
  {
    itkExceptionMacro(<< unimplementedOverrideMessage);
  }

  void
  GetSpatialHessian(const InputPointType &, SpatialHessianType &) const override
  {
    itkExceptionMacro(<< unimplementedOverrideMessage);
  }

  void
  GetJacobianOfSpatialJacobian(const InputPointType &,
                               JacobianOfSpatialJacobianType &,
                               NonZeroJacobianIndicesType &) const override
  {
    itkExceptionMacro(<< unimplementedOverrideMessage);
  }

  void
  GetJacobianOfSpatialJacobian(const InputPointType &,
                               SpatialJacobianType &,
                               JacobianOfSpatialJacobianType &,
                               NonZeroJacobianIndicesType &) const override
  {
    itkExceptionMacro(<< unimplementedOverrideMessage);
  }

  void
  GetJacobianOfSpatialHessian(const InputPointType &,
                              JacobianOfSpatialHessianType &,
                              NonZeroJacobianIndicesType &) const override
  {
    itkExceptionMacro(<< unimplementedOverrideMessage);
  }

  void
  GetJacobianOfSpatialHessian(const InputPointType &,
                              SpatialHessianType &,
                              JacobianOfSpatialHessianType &,
                              NonZeroJacobianIndicesType &) const override
  {
    itkExceptionMacro(<< unimplementedOverrideMessage);
  }

  using typename Superclass::TransformType;
  itkSetObjectMacro(ExternalTransform, TransformType);

  /** \note `GetModifiableExternalTransform()` is `const`, because it does not affect the adapter itself. */
  TransformType *
  GetModifiableExternalTransform() const
  {
    return m_ExternalTransform.GetPointer();
  }

protected:
  /** Default-constructor. */
  AdvancedTransformAdapter() = default;

  /** Destructor. */
  ~AdvancedTransformAdapter() override = default;

  /** Print contents of an AdvancedTransformAdapter. */
  void
  PrintSelf(std::ostream & os, itk::Indent indent) const override
  {
    Superclass::PrintSelf(os, indent);

    os << indent << "ExternalTransform: ";

    if (m_ExternalTransform)
    {
      os << *m_ExternalTransform << std::endl;
    }
    else
    {
      os << indent << "null" << std::endl;
    }
  }

private:
  // Private using-declarations, to avoid `-Woverloaded-virtual` warnings from GCC (GCC 11.4).
  using Superclass::TransformCovariantVector;
  using Superclass::TransformVector;

  static constexpr const char * unimplementedOverrideMessage = "Not implemented for AdvancedTransformAdapter";

  itk::SmartPointer<TransformType> m_ExternalTransform{};
};

} // namespace elastix


#endif /* itkAdvancedTransformAdapter_h */

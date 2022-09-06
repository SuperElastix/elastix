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
#ifndef itkStackTransform_h
#define itkStackTransform_h

#include "itkAdvancedTransform.h"
#include "itkIndex.h"

namespace itk
{

/** \class StackTransform
 * \brief Implements stack of transforms: one for every last dimension index.
 *
 * A list of transforms with dimension of Dimension - 1 is maintained:
 * one for every last dimension index. This transform selects the right
 * transform based on the last dimension index of the input point.
 *
 * \ingroup Transforms
 *
 */
template <class TScalarType, unsigned int NInputDimensions, unsigned int NOutputDimensions>
class ITK_TEMPLATE_EXPORT StackTransform : public AdvancedTransform<TScalarType, NInputDimensions, NOutputDimensions>
{
public:
  ITK_DISALLOW_COPY_AND_MOVE(StackTransform);

  /** Standard class typedefs. */
  using Self = StackTransform;
  using Superclass = AdvancedTransform<TScalarType, NInputDimensions, NOutputDimensions>;
  using Pointer = SmartPointer<Self>;
  using ConstPointer = SmartPointer<const Self>;

  /** Run-time type information (and related methods). */
  itkTypeMacro(StackTransform, AdvancedTransform);

  /** (Reduced) dimension of the domain space. */
  itkStaticConstMacro(InputSpaceDimension, unsigned int, NInputDimensions);
  itkStaticConstMacro(OutputSpaceDimension, unsigned int, NOutputDimensions);
  itkStaticConstMacro(ReducedInputSpaceDimension, unsigned int, NInputDimensions - 1);
  itkStaticConstMacro(ReducedOutputSpaceDimension, unsigned int, NOutputDimensions - 1);

  /** Typedefs from the Superclass. */
  using typename Superclass::ScalarType;
  using typename Superclass::ParametersType;
  using typename Superclass::FixedParametersType;
  using typename Superclass::NumberOfParametersType;
  using typename Superclass::ParametersValueType;
  using typename Superclass::JacobianType;
  using typename Superclass::SpatialJacobianType;
  using typename Superclass::JacobianOfSpatialJacobianType;
  using typename Superclass::SpatialHessianType;
  using typename Superclass::JacobianOfSpatialHessianType;
  using typename Superclass::NonZeroJacobianIndicesType;
  using typename Superclass::InputPointType;
  using typename Superclass::InputVectorType;
  using typename Superclass::OutputVectorType;
  using typename Superclass::InputVnlVectorType;
  using typename Superclass::OutputVnlVectorType;
  using typename Superclass::OutputCovariantVectorType;
  using typename Superclass::InputCovariantVectorType;
  using typename Superclass::OutputPointType;
  using typename Superclass::OutputVectorPixelType;
  using typename Superclass::InputVectorPixelType;

  /** Sub transform types, having a reduced dimension. */
  using SubTransformType =
    AdvancedTransform<TScalarType, Self::ReducedInputSpaceDimension, Self::ReducedOutputSpaceDimension>;
  using SubTransformPointer = typename SubTransformType::Pointer;
  using SubTransformJacobianType = typename SubTransformType::JacobianType;

  /** Dimension - 1 point types. */
  using SubTransformInputPointType = typename SubTransformType::InputPointType;
  using SubTransformOutputPointType = typename SubTransformType::OutputPointType;

  /** Array type for parameter vector instantiation. */
  using ParametersArrayType = typename ParametersType::ArrayType;

  /**  Method to transform a point. */
  OutputPointType
  TransformPoint(const InputPointType & inputPoint) const override;

  /** This returns a sparse version of the Jacobian of the transformation.
   * In this class however, the Jacobian is not sparse.
   * However, it is a useful function, since the Jacobian is passed
   * by reference, which makes it threadsafe, unlike the normal
   * GetJacobian function. */
  void
  GetJacobian(const InputPointType & inputPoint, JacobianType & jac, NonZeroJacobianIndicesType & nzji) const override;

  /** Set the parameters. Checks if the number of parameters
   * is correct and sets parameters of sub transforms. */
  void
  SetParameters(const ParametersType & param) override;

  /** Get the parameters. Concatenates the parameters of the
   * sub transforms. */
  const ParametersType &
  GetParameters() const override;

  /** Set the fixed parameters. */
  void
  SetFixedParameters(const FixedParametersType & fixedParameters) override
  {
    const auto numberOfFixedParameters = fixedParameters.size();
    if (numberOfFixedParameters < NumberOfGeneralFixedParametersOfStack)
    {
      itkExceptionMacro(<< "The number of FixedParameters (" << numberOfFixedParameters << ") should be at least "
                        << NumberOfGeneralFixedParametersOfStack);
    }

    if (Superclass::m_FixedParameters != fixedParameters)
    {
      Superclass::m_FixedParameters = fixedParameters;

      CreateSubTransforms(FixedParametersType(fixedParameters.data_block() + NumberOfGeneralFixedParametersOfStack,
                                              numberOfFixedParameters - NumberOfGeneralFixedParametersOfStack));
      UpdateStackSpacingAndOrigin();
      this->Modified();
    }
  }


  /** Return the number of sub transforms that have been set. */
  NumberOfParametersType
  GetNumberOfParameters() const override
  {
    if (this->m_SubTransformContainer.empty())
    {
      return 0;
    }
    else
    {
      return this->m_SubTransformContainer.size() * m_SubTransformContainer[0]->GetNumberOfParameters();
    }
  }


  /** Set/get number of transforms needed. */
  void
  SetNumberOfSubTransforms(const unsigned int num)
  {
    if (this->m_SubTransformContainer.size() != num)
    {
      this->m_SubTransformContainer.clear();
      this->m_SubTransformContainer.resize(num);
      this->Modified();
    }
  }


  auto
  GetNumberOfSubTransforms() const
  {
    return static_cast<unsigned>(m_SubTransformContainer.size());
  }


  /** Set/get stack transform parameters. */
  itkSetMacro(StackSpacing, TScalarType);
  itkGetConstMacro(StackSpacing, TScalarType);
  itkSetMacro(StackOrigin, TScalarType);
  itkGetConstMacro(StackOrigin, TScalarType);

  /** Set the initial transform for sub transform i. */
  void
  SetSubTransform(unsigned int i, SubTransformType * transform)
  {
    this->m_SubTransformContainer[i] = transform;
    this->Modified();
  }


  /** Sets the fixed parameters to the general fixed parameters of the stack + the fixed parameters of the first
   * sub-transform (if any). */
  void
  UpdateFixedParameters()
  {
    const SubTransformType * const subTransform =
      m_SubTransformContainer.empty() ? nullptr : m_SubTransformContainer.front();
    this->UpdateFixedParametersInternally((subTransform == nullptr) ? FixedParametersType()
                                                                    : subTransform->GetFixedParameters());
  }


  /** Set all sub transforms to transform. */
  void
  SetAllSubTransforms(const SubTransformType & transform)
  {
    const auto & fixedParametersOfSubTransform = transform.GetFixedParameters();
    const auto & parametersOfSubTransform = transform.GetParameters();

    UpdateFixedParametersInternally(fixedParametersOfSubTransform);

    for (auto & subTransform : m_SubTransformContainer)
    {
      // Copy transform
      SubTransformPointer transformcopy = dynamic_cast<SubTransformType *>(transform.CreateAnother().GetPointer());
      transformcopy->SetFixedParameters(fixedParametersOfSubTransform);
      transformcopy->SetParameters(parametersOfSubTransform);
      // Set sub transform
      subTransform = transformcopy;
    }
  }


  /** Get a sub transform. */
  SubTransformPointer
  GetSubTransform(unsigned int i)
  {
    return this->m_SubTransformContainer[i];
  }


  /** Get number of nonzero Jacobian indices. */
  NumberOfParametersType
  GetNumberOfNonZeroJacobianIndices() const override;

protected:
  StackTransform() = default;
  ~StackTransform() override = default;

  // Indices of the general fixed parameters into the FixedParameters array, and the number of those parameters.
  enum
  {
    IndexOfNumberOfSubTransforms,
    IndexOfStackSpacing,
    IndexOfStackOrigin,
    NumberOfGeneralFixedParametersOfStack
  };

  void
  CreateSubTransforms(const FixedParametersType & fixedParametersOfSubTransform)
  {
    assert(Superclass::m_FixedParameters.size() >= NumberOfGeneralFixedParametersOfStack);
    const auto numberOfSubTransforms = Superclass::m_FixedParameters[IndexOfNumberOfSubTransforms];

    if (numberOfSubTransforms >= 0.0 && numberOfSubTransforms <= UINT_MAX &&
        static_cast<double>(static_cast<unsigned>(numberOfSubTransforms)) == numberOfSubTransforms)
    {
      m_SubTransformContainer.resize(static_cast<unsigned>(numberOfSubTransforms));
    }
    else
    {
      itkExceptionMacro(<< "The FixedParameters element (" << numberOfSubTransforms
                        << ") should be a valid number (the number of subtransforms).");
    }

    for (auto & subTransform : m_SubTransformContainer)
    {
      subTransform = this->CreateSubTransform();
      subTransform->SetFixedParameters(fixedParametersOfSubTransform);
    }
  }

  void
  UpdateStackSpacingAndOrigin()
  {
    assert(Superclass::m_FixedParameters.size() >= NumberOfGeneralFixedParametersOfStack);
    m_StackSpacing = Superclass::m_FixedParameters[IndexOfStackSpacing];
    m_StackOrigin = Superclass::m_FixedParameters[IndexOfStackOrigin];
  }


  /** Sets the fixed parameters to the general fixed parameters of the stack + the specified fixed parameters of a
   * sub-transform. */
  virtual void
  UpdateFixedParametersInternally(const FixedParametersType & fixedParametersOfSubTransform)
  {
    const auto numberOfFixedParametersOfSubTransform = fixedParametersOfSubTransform.size();

    FixedParametersType & fixedParametersOfStack = this->Superclass::m_FixedParameters;

    const auto minimumNumberOfFixedParametersOfStack =
      NumberOfGeneralFixedParametersOfStack + numberOfFixedParametersOfSubTransform;

    if (fixedParametersOfStack.size() < minimumNumberOfFixedParametersOfStack)
    {
      fixedParametersOfStack.set_size(minimumNumberOfFixedParametersOfStack);
    }
    fixedParametersOfStack[IndexOfNumberOfSubTransforms] = m_SubTransformContainer.size();
    fixedParametersOfStack[IndexOfStackOrigin] = m_StackOrigin;
    fixedParametersOfStack[IndexOfStackSpacing] = m_StackSpacing;
    std::copy_n(fixedParametersOfSubTransform.begin(),
                numberOfFixedParametersOfSubTransform,
                fixedParametersOfStack.begin() + NumberOfGeneralFixedParametersOfStack);
  }

private:
  /** Each override of this pure virtual member function should create a subtransform for the specific (derived) stack
   * transform type. For example, for an `TranslationStackTransform` it should create an `AdvancedTranslationTransform`,
   * and for an `EulerStackTransform` it should create an `EulerTransform`. */
  virtual SubTransformPointer
  CreateSubTransform() const = 0;


  static constexpr const char * unimplementedOverrideMessage = "Not implemented for StackTransform";

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


  /** Must be provided. */
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


  // Transform container
  std::vector<SubTransformPointer> m_SubTransformContainer;

  // Stack spacing and origin of last dimension
  TScalarType m_StackSpacing{ 1.0 };
  TScalarType m_StackOrigin{ 0.0 };
};

} // end namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#  include "itkStackTransform.hxx"
#endif

#endif

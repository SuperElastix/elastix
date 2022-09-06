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
#ifndef itkEulerTransform_h
#define itkEulerTransform_h

#include "itkAdvancedRigid2DTransform.h"
#include "itkAdvancedEuler3DTransform.h"
#include "itkAdvancedMatrixOffsetTransformBase.h"

namespace itk
{

/**
 * \class EulerGroup
 * \brief This class only contains an alias template.
 *
 */

template <unsigned int Dimension>
class ITK_TEMPLATE_EXPORT EulerGroup
{
public:
  template <class TScalarType>
  using TransformAlias = AdvancedMatrixOffsetTransformBase<TScalarType, Dimension, Dimension>;
};

/**
 * \class EulerGroup<2>
 * \brief This class only contains an alias template for the 2D case.
 *
 */

template <>
class ITK_TEMPLATE_EXPORT EulerGroup<2>
{
public:
  template <class TScalarType>
  using TransformAlias = AdvancedRigid2DTransform<TScalarType>;
};

/**
 * \class EulerGroup<3>
 * \brief This class only contains an alias template for the 3D case.
 *
 */

template <>
class ITK_TEMPLATE_EXPORT EulerGroup<3>
{
public:
  template <class TScalarType>
  using TransformAlias = AdvancedEuler3DTransform<TScalarType>;
};


/**
 * This alias templates the EulerGroup over its dimension.
 */
template <class TScalarType, unsigned int Dimension>
using EulerGroupTemplate = typename EulerGroup<Dimension>::template TransformAlias<TScalarType>;


/**
 * \class EulerTransform
 * \brief This class combines the Euler2DTransform with the Euler3DTransform.
 *
 * This transform is a rigid body transformation.
 *
 * \ingroup Transforms
 */

template <class TScalarType, unsigned int Dimension>
class ITK_TEMPLATE_EXPORT EulerTransform : public EulerGroupTemplate<TScalarType, Dimension>
{
public:
  ITK_DISALLOW_COPY_AND_MOVE(EulerTransform);

  /** Standard ITK-stuff. */
  using Self = EulerTransform;
  using Superclass = EulerGroupTemplate<TScalarType, Dimension>;
  using Pointer = SmartPointer<Self>;
  using ConstPointer = SmartPointer<const Self>;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Run-time type information (and related methods). */
  itkTypeMacro(EulerTransform, EulerGroupTemplate);

  /** Dimension of the domain space. */
  itkStaticConstMacro(SpaceDimension, unsigned int, Dimension);

  /** Typedefs inherited from the superclass. */

  /** These are both in Rigid2D and Euler3D. */
  using typename Superclass::ScalarType;
  using typename Superclass::ParametersType;
  using typename Superclass::NumberOfParametersType;
  using typename Superclass::JacobianType;
  using typename Superclass::OffsetType;
  using typename Superclass::InputPointType;
  using typename Superclass::OutputPointType;
  using typename Superclass::InputVectorType;
  using typename Superclass::OutputVectorType;
  using typename Superclass::InputCovariantVectorType;
  using typename Superclass::OutputCovariantVectorType;
  using typename Superclass::InputVnlVectorType;
  using typename Superclass::OutputVnlVectorType;

  using typename Superclass::NonZeroJacobianIndicesType;
  using typename Superclass::SpatialJacobianType;
  using typename Superclass::JacobianOfSpatialJacobianType;
  using typename Superclass::SpatialHessianType;
  using typename Superclass::JacobianOfSpatialHessianType;
  using typename Superclass::InternalMatrixType;

  /** Make sure SetComputeZYX() is available, also in 2D,
   * in which case, its just a dummy function.
   */
  void
  SetComputeZYX(const bool) // No override.
  {
    static_assert(SpaceDimension != 3, "This is not the specialization is 3D!");
  }


  /** Make sure GetComputeZYX() is available, also in 2D,
   * in which case, it just returns false.
   */
  bool
  GetComputeZYX() const // No override.
  {
    static_assert(SpaceDimension != 3, "This is not the specialization is 3D!");
    return false;
  }


protected:
  EulerTransform() = default;
  ~EulerTransform() override = default;
};

template <class TScalarType>
class ITK_TEMPLATE_EXPORT EulerTransform<TScalarType, 3> : public EulerGroupTemplate<TScalarType, 3>
{
public:
  ITK_DISALLOW_COPY_AND_MOVE(EulerTransform);

  /** Standard ITK-stuff. */
  using Self = EulerTransform;
  using Superclass = EulerGroupTemplate<TScalarType, 3>;
  using Pointer = SmartPointer<Self>;
  using ConstPointer = SmartPointer<const Self>;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Run-time type information (and related methods). */
  itkTypeMacro(EulerTransform, EulerGroupTemplate);

  /** Dimension of the domain space. */
  itkStaticConstMacro(SpaceDimension, unsigned int, 3);


  /** Make sure SetComputeZYX() is available, also in 2D,
   * in which case, its just a dummy function.
   * \note This member function is only an `override` in 3D.
   */
  void
  SetComputeZYX(const bool arg) override
  {
    static_assert(SpaceDimension == 3, "This specialization is for 3D only!");

    using Euler3DTransformType = AdvancedEuler3DTransform<TScalarType>;
    typename Euler3DTransformType::Pointer transform = dynamic_cast<Euler3DTransformType *>(this);
    if (transform)
    {
      transform->Euler3DTransformType::SetComputeZYX(arg);
    }
  }


  /** Make sure GetComputeZYX() is available, also in 2D,
   * in which case, it just returns false.
   * \note This member function is only an `override` in 3D.
   */
  bool
  GetComputeZYX() const override
  {
    static_assert(SpaceDimension == 3, "This specialization is for 3D only!");

    using Euler3DTransformType = AdvancedEuler3DTransform<TScalarType>;
    typename Euler3DTransformType::ConstPointer transform = dynamic_cast<const Euler3DTransformType *>(this);

    if (transform)
    {
      return transform->Euler3DTransformType::GetComputeZYX();
    }
    return false;
  }


protected:
  EulerTransform() = default;
  ~EulerTransform() override = default;
};

} // end namespace itk

#endif // end #ifndef itkEulerTransform_h

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
#ifndef __itkLimitedEulerTransform_H__
#define __itkLimitedEulerTransform_H__

#include "itkAdvancedRigid2DTransform.h"
#include "itkAdvancedLimitedEuler3DTransform.h"
#include "itkAdvancedMatrixOffsetTransformBase.h"

namespace itk
{

  /**
   * \class LimitedEulerGroup
   * \brief This class only contains a dummy class.
   *
   */

  template< unsigned int Dimension >
  class LimitedEulerGroup
  {
  public:

    template< class TScalarType >
    class Dummy
    {
    public:

      /** Typedef's.*/
      typedef AdvancedMatrixOffsetTransformBase< TScalarType, Dimension, Dimension > LimitedEulerTransform_tmp;

    };

  };

  /**
   * \class LimitedEulerGroup<2>
   * \brief This class only contains a dummy class for the 2D case.
   *
   */

  template< >
  class LimitedEulerGroup< 2 >
  {
  public:

    template< class TScalarType >
    class Dummy
    {
    public:

      /** Typedef's.*/
      typedef AdvancedRigid2DTransform< TScalarType > LimitedEulerTransform_tmp;

    };

  };

  /**
   * \class LimitedEulerGroup<3>
   * \brief This class only contains a dummy class for the 3D case.
   *
   */

  template< >
  class LimitedEulerGroup< 3 >
  {
  public:

    template< class TScalarType >
    class Dummy
    {
    public:

      /** Typedef's.*/
      typedef AdvancedLimitedEuler3DTransform< TScalarType > LimitedEulerTransform_tmp;
    };

  };

  /**
   * \class LimitedEulerGroupTemplate
   * \brief This class templates the EulerGroup over its dimension.
   *
   */

  template< class TScalarType, unsigned int Dimension >
  class LimitedEulerGroupTemplate
  {
  public:

    typedef LimitedEulerGroupTemplate Self;
    typedef TScalarType        ScalarType;
    itkStaticConstMacro(SpaceDimension, unsigned int, Dimension);

    // This declaration of 'LimitedEuler' does not work with the GCC compiler
    //    typedef LimitedEulerGroup<  itkGetStaticConstMacro( SpaceDimension ) >       LimitedEuler;
    // The following trick works though:
    template< unsigned int D >
    class LimitedEulerGroupWrap
    {
    public:

      typedef LimitedEulerGroup< D > Euler;
    };

    typedef LimitedEulerGroupWrap< Dimension >            LimitedEulerGroupWrapInstance;
    typedef typename LimitedEulerGroupWrapInstance::Euler LimitedEuler;

    typedef typename LimitedEuler::template Dummy< ScalarType > LimitedEulerDummy;
    typedef typename LimitedEulerDummy::LimitedEulerTransform_tmp     LimitedEulerTransform_tmp;

  };

  /**
   * \class LimitedEulerTransform
   * \brief This class combines the Euler2DTransform with the Euler3DTransform.
   *
   * This transform is a rigid body transformation.
   *
   * \ingroup Transforms
   */

  template< class TScalarType, unsigned int Dimension >
  class LimitedEulerTransform :
    public LimitedEulerGroupTemplate<
    TScalarType, Dimension >::LimitedEulerTransform_tmp
  {
  public:

    /** Standard ITK-stuff. */
    typedef LimitedEulerTransform Self;
    typedef typename LimitedEulerGroupTemplate<
      TScalarType, Dimension >
      ::LimitedEulerTransform_tmp Superclass;
    typedef SmartPointer< Self >       Pointer;
    typedef SmartPointer< const Self > ConstPointer;

    /** Method for creation through the object factory. */
    itkNewMacro(Self);

    /** Run-time type information (and related methods). */
    itkTypeMacro(LimitedEulerTransform, LimitedEulerGroupTemplate);

    /** Dimension of the domain space. */
    itkStaticConstMacro(SpaceDimension, unsigned int, Dimension);

    /** Typedefs inherited from the superclass. */

    /** These are both in Rigid2D and Euler3D. */
    typedef typename Superclass::ScalarType                ScalarType;
    typedef typename Superclass::ParametersType            ParametersType;
    typedef typename Superclass::NumberOfParametersType    NumberOfParametersType;
    typedef typename Superclass::JacobianType              JacobianType;
    typedef typename Superclass::OffsetType                OffsetType;
    typedef typename Superclass::InputPointType            InputPointType;
    typedef typename Superclass::OutputPointType           OutputPointType;
    typedef typename Superclass::InputVectorType           InputVectorType;
    typedef typename Superclass::OutputVectorType          OutputVectorType;
    typedef typename Superclass::InputCovariantVectorType  InputCovariantVectorType;
    typedef typename Superclass::OutputCovariantVectorType OutputCovariantVectorType;
    typedef typename Superclass::InputVnlVectorType        InputVnlVectorType;
    typedef typename Superclass::OutputVnlVectorType       OutputVnlVectorType;

    typedef typename Superclass
      ::NonZeroJacobianIndicesType NonZeroJacobianIndicesType;
    typedef typename Superclass::SpatialJacobianType SpatialJacobianType;
    typedef typename Superclass
      ::JacobianOfSpatialJacobianType JacobianOfSpatialJacobianType;
    typedef typename Superclass::SpatialHessianType SpatialHessianType;
    typedef typename Superclass
      ::JacobianOfSpatialHessianType JacobianOfSpatialHessianType;
    typedef typename Superclass::InternalMatrixType InternalMatrixType;

    /** Make sure SetComputeZYX() is available, also in 2D,
     * in which case, its just a dummy function.
     */
    void SetComputeZYX(const bool) // No override.
    {
      static_assert(SpaceDimension != 3, "This is not the specialization is 3D!");
    }

    /** Make sure GetComputeZYX() is available, also in 2D,
     * in which case, it just returns false.
     */
    bool GetComputeZYX(void) const // No override.
    {
      static_assert(SpaceDimension != 3, "This is not the specialization is 3D!");
      return false;
    }

    /** Make sure setters/getters are available, also in 2D,
     * in which case, it just issues an assert error.
     */
    void SetScalesEstimation(const bool arg) // No override.
    {
      static_assert(SpaceDimension != 3, "Not implemented in 2D!");
    }

    void SetSharpnessOfLimits(const ScalarType sharpnessOfLimits) // No override.
    {
      static_assert(SpaceDimension != 3, "Not implemented in 2D!");
    }

    const ScalarType& GetSharpnessOfLimits() // No override.
    {
      static_assert(SpaceDimension != 3, "Not implemented in 2D!");
      return ScalarType();
    }

    void SetUpperLimits(const ParametersType & upperLimits) // No override.
    {
      static_assert(SpaceDimension != 3, "Not implemented in 2D!");
    }

    const ParametersType& GetUpperLimits()
    {
      static_assert(SpaceDimension != 3, "Not implemented in 2D!");
      return ParametersType();
    }

    void SetLowerLimits(const ParametersType & lowerLimits)
    {
      static_assert(SpaceDimension != 3, "Not implemented in 2D!");
    }

    const ParametersType& GetLowerLimits()
    {
      static_assert(SpaceDimension != 3, "Not implemented in 2D!");
      return ParametersType();
    }

    const ParametersType& GetUpperLimitsReached()
    {
      static_assert(SpaceDimension != 3, "Not implemented in 2D!");
      return ParametersType();
    }

    const ParametersType& GetLowerLimitsReached()
    {
      static_assert(SpaceDimension != 3, "Not implemented in 2D!");
      return ParametersType();
    }

  protected:

    LimitedEulerTransform() {}
    ~LimitedEulerTransform() override {}

  private:

    LimitedEulerTransform(const Self&);  // purposely not implemented
    void operator=(const Self&);  // purposely not implemented

  };

  template< class TScalarType >
  class LimitedEulerTransform<TScalarType, 3> :
    public LimitedEulerGroupTemplate<
    TScalarType, 3 >::LimitedEulerTransform_tmp
  {
  public:

    /** Standard ITK-stuff. */
    typedef LimitedEulerTransform Self;
    typedef typename LimitedEulerGroupTemplate<
      TScalarType, 3 >
      ::LimitedEulerTransform_tmp Superclass;
    typedef SmartPointer< Self >       Pointer;
    typedef SmartPointer< const Self > ConstPointer;

    typedef typename Superclass::ScalarType                ScalarType;
    typedef typename Superclass::ParametersType            ParametersType;

    /** Method for creation through the object factory. */
    itkNewMacro(Self);

    /** Run-time type information (and related methods). */
    itkTypeMacro(LimitedEulerTransform, LimitedEulerGroupTemplate);

    /** Dimension of the domain space. */
    itkStaticConstMacro(SpaceDimension, unsigned int, 3);


    /** Make sure SetComputeZYX() is available, also in 2D,
     * in which case, its just a dummy function.
     * \note This member function is only an `override` in 3D.
     */
    void SetComputeZYX(const bool arg) override
    {
      static_assert(SpaceDimension == 3, "This specialization is for 3D only!");

      typedef AdvancedLimitedEuler3DTransform< TScalarType > Euler3DTransformType;
      typename Euler3DTransformType::Pointer transform
        = dynamic_cast<Euler3DTransformType*>(this);
      if (transform)
      {
        transform->Euler3DTransformType::SetComputeZYX(arg);
      }
    }


    /** Make sure GetComputeZYX() is available, also in 2D,
     * in which case, it just returns false.
     * \note This member function is only an `override` in 3D.
     */
    bool GetComputeZYX(void) const override
    {
      static_assert(SpaceDimension == 3, "This specialization is for 3D only!");

      typedef AdvancedLimitedEuler3DTransform< TScalarType > Euler3DTransformType;
      typename Euler3DTransformType::ConstPointer transform
        = dynamic_cast<const Euler3DTransformType*>(this);

      if (transform)
      {
        return transform->Euler3DTransformType::GetComputeZYX();
      }
      return false;
    }


    /** Make sure setters/getters are available, also in 3D,
     * in which case, it sets/returns values accordingly.
     */
    void SetScalesEstimation(const bool arg) override
    {
      static_assert(SpaceDimension == 3, "This specialization is for 3D only!");

      typedef AdvancedLimitedEuler3DTransform< TScalarType > Euler3DTransformType;
      typename Euler3DTransformType::Pointer transform
        = dynamic_cast<Euler3DTransformType*>(this);

      if (transform)
      {
        return transform->Euler3DTransformType::SetScalesEstimation(arg);
      }
    }

    void SetSharpnessOfLimits(const ScalarType sharpnessOfLimits)
    {
      static_assert(SpaceDimension == 3, "This specialization is for 3D only!");

      typedef AdvancedLimitedEuler3DTransform< TScalarType > Euler3DTransformType;
      typename Euler3DTransformType::Pointer transform
        = dynamic_cast< Euler3DTransformType * >( this );

      if( transform )
      {
        transform->Euler3DTransformType::SetSharpnessOfLimits(sharpnessOfLimits);
        transform->Euler3DTransformType::UpdateSharpnessOfLimitsVector();
      }
    }

    const ScalarType& GetSharpnessOfLimits() // const override
    {
      static_assert(SpaceDimension == 3, "This specialization is for 3D only!");

      typedef AdvancedLimitedEuler3DTransform< TScalarType > Euler3DTransformType;
      typename Euler3DTransformType::Pointer transform
        = dynamic_cast< Euler3DTransformType * >( this );

      if( transform )
      {
        return transform->Euler3DTransformType::GetSharpnessOfLimits();
      }
      return ScalarType();
    }

    void SetUpperLimits(const ParametersType & upperLimits) override
    {
      static_assert(SpaceDimension == 3, "This specialization is for 3D only!");

      typedef AdvancedLimitedEuler3DTransform< TScalarType > Euler3DTransformType;
      typename Euler3DTransformType::Pointer transform
        = dynamic_cast< Euler3DTransformType * >( this );

      if( transform )
      {
        transform->Euler3DTransformType::SetUpperLimits( upperLimits );
      }
    }

    const ParametersType& GetUpperLimits() //const override
    {
      static_assert(SpaceDimension == 3, "This specialization is for 3D only!");

      typedef AdvancedLimitedEuler3DTransform< TScalarType > Euler3DTransformType;
      typename Euler3DTransformType::Pointer transform
        = dynamic_cast< Euler3DTransformType * >( this );

      if( transform )
      {
        return transform->Euler3DTransformType::GetUpperLimits();
      }
      return ParametersType();
    }

    void SetLowerLimits(const ParametersType & lowerLimits) override
    {
      static_assert(SpaceDimension == 3, "This specialization is for 3D only!");

      typedef AdvancedLimitedEuler3DTransform< TScalarType > Euler3DTransformType;
      typename Euler3DTransformType::Pointer transform
        = dynamic_cast< Euler3DTransformType * >( this );

      if( transform )
      {
        transform->Euler3DTransformType::SetLowerLimits( lowerLimits );
      }
    }

    const ParametersType& GetLowerLimits() //const override
    {
      static_assert(SpaceDimension == 3, "This specialization is for 3D only!");

      typedef AdvancedLimitedEuler3DTransform< TScalarType > Euler3DTransformType;
      typename Euler3DTransformType::Pointer transform
        = dynamic_cast< Euler3DTransformType * >( this );

      if( transform )
      {
        return transform->Euler3DTransformType::GetLowerLimits();
      }
      return ParametersType();
    }

    const ParametersType& GetUpperLimitsReached() //const override
    {
      static_assert(SpaceDimension == 3, "This specialization is for 3D only!");

      typedef AdvancedLimitedEuler3DTransform< TScalarType > Euler3DTransformType;
      typename Euler3DTransformType::Pointer transform
        = dynamic_cast<Euler3DTransformType*>(this);

      if (transform)
      {
        return transform->Euler3DTransformType::GetUpperLimitsReached();
      }
      return ParametersType();
    }

    const ParametersType& GetLowerLimitsReached() //const override
    {
      static_assert(SpaceDimension == 3, "This specialization is for 3D only!");

      typedef AdvancedLimitedEuler3DTransform< TScalarType > Euler3DTransformType;
      typename Euler3DTransformType::Pointer transform
        = dynamic_cast<Euler3DTransformType*>(this);

      if (transform)
      {
        return transform->Euler3DTransformType::GetLowerLimitsReached();
      }
      return ParametersType();
    }

  protected:

    LimitedEulerTransform() {}
    ~LimitedEulerTransform() override {}

  private:

    LimitedEulerTransform(const Self&);  // purposely not implemented
    void operator=(const Self&);  // purposely not implemented
  };

} // end namespace itk

#endif // end #ifndef __itkLimitedEulerTransform_H__

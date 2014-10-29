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
#ifndef __itkAffineDTITransform_H__
#define __itkAffineDTITransform_H__

#include "itkAffineDTI2DTransform.h"
#include "itkAffineDTI3DTransform.h"
#include "itkAdvancedMatrixOffsetTransformBase.h"

namespace itk
{

/**
 * \class AffineDTIGroup
 * \brief This class only contains a dummy class.
 *
 */

template< unsigned int Dimension >
class AffineDTIGroup
{
public:

  template< class TScalarType >
  class Dummy
  {
public:

    /** Typedef's.*/
    typedef AdvancedMatrixOffsetTransformBase< TScalarType, Dimension, Dimension > AffineDTITransform_tmp;

  };

};
/**
 * \class AffineDTIGroup<2>
 * \brief This class only contains a dummy class for the 2D case.
 *
 */

template< >
class AffineDTIGroup< 2 >
{
public:

  template< class TScalarType >
  class Dummy
  {
public:

    /** Typedef's.*/
    typedef AffineDTI2DTransform< TScalarType > AffineDTITransform_tmp;

  };

};
/**
 * \class AffineDTIGroup<3>
 * \brief This class only contains a dummy class for the 3D case.
 *
 */

template< >
class AffineDTIGroup< 3 >
{
public:

  template< class TScalarType >
  class Dummy
  {
public:

    /** Typedef's.*/
    typedef AffineDTI3DTransform< TScalarType > AffineDTITransform_tmp;

  };

};

/**
 * \class AffineDTIGroupTemplate
 * \brief This class templates the AffineDTIGroup over its dimension.
 *
 */

template< class TScalarType, unsigned int Dimension >
class AffineDTIGroupTemplate
{
public:

  typedef AffineDTIGroupTemplate Self;
  typedef TScalarType            ScalarType;
  itkStaticConstMacro( SpaceDimension, unsigned int, Dimension );

  // This declaration of 'AffineDTI' does not work with the GCC compiler
  //    typedef AffineDTIGroup<  itkGetStaticConstMacro( SpaceDimension ) >       AffineDTI;
  // The following trick works though:
  template< unsigned int D >
  class AffineDTIGroupWrap
  {
public:

    typedef AffineDTIGroup< D > AffineDTI;
  };

  typedef AffineDTIGroupWrap< Dimension >                AffineDTIGroupWrapInstance;
  typedef typename AffineDTIGroupWrapInstance::AffineDTI AffineDTI;

  typedef typename AffineDTI::template Dummy< ScalarType > AffineDTIDummy;
  typedef typename AffineDTIDummy::AffineDTITransform_tmp  AffineDTITransform_tmp;

};

/**
 * \class AffineDTITransform
 * \brief This class combines the AffineDTI2DTransform with the AffineDTI3DTransform.
 *
 * This transform is an affine transform with MR-DTI specific parametrization.
 *
 * \ingroup Transforms
 */

template< class TScalarType, unsigned int Dimension >
class AffineDTITransform :
  public AffineDTIGroupTemplate<
  TScalarType, Dimension >::AffineDTITransform_tmp
{
public:

  /** Standard ITK-stuff. */
  typedef AffineDTITransform Self;
  typedef typename AffineDTIGroupTemplate<
    TScalarType, Dimension >
    ::AffineDTITransform_tmp Superclass;
  typedef SmartPointer< Self >       Pointer;
  typedef SmartPointer< const Self > ConstPointer;

  /** Method for creation through the object factory. */
  itkNewMacro( Self );

  /** Run-time type information (and related methods). */
  itkTypeMacro( AffineDTITransform, AffineDTIGroupTemplate );

  /** Dimension of the domain space. */
  itkStaticConstMacro( SpaceDimension, unsigned int, Dimension );

  /** Typedefs inherited from the superclass. */

  /** These are both in AffineDTI2D and AffineDTI3D. */
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

protected:

  AffineDTITransform(){}
  ~AffineDTITransform(){}

private:

  AffineDTITransform( const Self & ); // purposely not implemented
  void operator=( const Self & );     // purposely not implemented

};

} // end namespace itk

#endif // end #ifndef __itkAffineDTITransform_H__

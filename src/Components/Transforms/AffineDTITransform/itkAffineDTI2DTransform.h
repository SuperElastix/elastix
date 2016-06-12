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
#ifndef __itkAffineDTI2DTransform_h
#define __itkAffineDTI2DTransform_h

#include <iostream>
#include "itkAdvancedMatrixOffsetTransformBase.h"

namespace itk
{

/** \class AffineDTI3DTransform
 *
 * \brief AffineDTI3DTransform of a vector space (e.g. space coordinates)
 *
 * This transform applies an affine transformation, but is parameterized by
 * angles, shear factors, scales, and translation, instead of by the affine matrix.
 * It is meant for registration of MR diffusion weighted images, but could be
 * used for other images as well of course.
 *
 * The affine model is adopted from the following paper:
 * [1] A. Leemans and D.K. Jones. "The B-matrix must be rotated when correcting for subject motion in DTI data".
 *   Magnetic Resonance in Medicine, Volume 61, Issue 6, pages 1336 - 1349, 2009.
 *
 * The model is as follows:\n
 *   T(x) = R G S (x-c) + t + c\n
 * with:
 *   - R = Rx Ry Rz (rotation matrices)
 *   - G = Gx Gy Gz (shear matrices)
 *   - S = diag( [sx sy sz] ) (scaling matrix)
 *   - c = center of rotation
 *   - t = translation
 * See [1] for exact expressions for Rx, Gx etc.
 *
 * Using this model, the rotation components can be easily extracted an applied
 * to the B-matrix.
 *
 * The parameters are ordered as follows:
 * <tt>in 2D: [ Angle ShearX ShearY ScaleX ScaleY TranslationX TranslationY ]</tt>
 * <tt>in 3D: [ AngleX AngleY AngleZ ShearX ShearY ShearZ ScaleX ScaleY ScaleZ TranslationX TranslationY TranslationZ ]</tt>
 *
 * The serialization of the fixed parameters is an array of 2 elements defining
 * the center of rotation.
 *
 * \ingroup Transforms
 */
template< class TScalarType = double >
// Data type for scalars (float or double)
class AffineDTI2DTransform :
  public AdvancedMatrixOffsetTransformBase< TScalarType, 2, 2 >
{
public:

  /** Standard class typedefs. */
  typedef AffineDTI2DTransform                                   Self;
  typedef AdvancedMatrixOffsetTransformBase< TScalarType, 2, 2 > Superclass;
  typedef SmartPointer< Self >                                   Pointer;
  typedef SmartPointer< const Self >                             ConstPointer;

  /** New macro for creation of through a Smart Pointer. */
  itkNewMacro( Self );

  /** Run-time type information (and related methods). */
  itkTypeMacro( AffineDTI2DTransform, AdvancedMatrixOffsetTransformBase );

  /** Dimension of the space. */
  itkStaticConstMacro( SpaceDimension, unsigned int, 2 );
  itkStaticConstMacro( InputSpaceDimension, unsigned int, 2 );
  itkStaticConstMacro( OutputSpaceDimension, unsigned int, 2 );
  itkStaticConstMacro( ParametersDimension, unsigned int, 7 );

  typedef typename Superclass::ParametersType            ParametersType;
  typedef typename Superclass::NumberOfParametersType    NumberOfParametersType;
  typedef typename Superclass::JacobianType              JacobianType;
  typedef typename Superclass::ScalarType                ScalarType;
  typedef typename Superclass::InputVectorType           InputVectorType;
  typedef typename Superclass::OutputVectorType          OutputVectorType;
  typedef typename Superclass::InputCovariantVectorType  InputCovariantVectorType;
  typedef typename Superclass::OutputCovariantVectorType OutputCovariantVectorType;
  typedef typename Superclass::InputVnlVectorType        InputVnlVectorType;
  typedef typename Superclass::OutputVnlVectorType       OutputVnlVectorType;
  typedef typename Superclass::InputPointType            InputPointType;
  typedef typename Superclass::OutputPointType           OutputPointType;
  typedef typename Superclass::MatrixType                MatrixType;
  typedef typename Superclass::InverseMatrixType         InverseMatrixType;
  typedef typename Superclass::CenterType                CenterType;
  typedef typename Superclass::TranslationType           TranslationType;
  typedef typename Superclass::OffsetType                OffsetType;
  typedef typename Superclass::ScalarType                AngleType;

  typedef typename Superclass
    ::NonZeroJacobianIndicesType NonZeroJacobianIndicesType;
  typedef typename Superclass::SpatialJacobianType SpatialJacobianType;
  typedef typename Superclass
    ::JacobianOfSpatialJacobianType JacobianOfSpatialJacobianType;
  typedef typename Superclass::SpatialHessianType SpatialHessianType;
  typedef typename Superclass
    ::JacobianOfSpatialHessianType JacobianOfSpatialHessianType;
  typedef typename Superclass::InternalMatrixType InternalMatrixType;

  typedef FixedArray< ScalarType > ScalarArrayType;

  /** Set/Get the transformation from a container of parameters
   * This is typically used by optimizers.  There are 7 parameters.
   * [ R Gx Gy Sx Sy Tx Ty ]
   * ~rotation, scale, skew, translation
   */
  void SetParameters( const ParametersType & parameters );

  const ParametersType & GetParameters( void ) const;

  /** Compute the Jacobian of the transformation. */
  virtual void GetJacobian(
    const InputPointType &,
    JacobianType &,
    NonZeroJacobianIndicesType & ) const;

  virtual void SetIdentity( void );

protected:

  AffineDTI2DTransform();
  AffineDTI2DTransform( const MatrixType & matrix,
    const OutputPointType & offset );
  AffineDTI2DTransform( unsigned int outputSpaceDims,
    unsigned int paramsSpaceDims );

  ~AffineDTI2DTransform(){}

  void PrintSelf( std::ostream & os, Indent indent ) const;

  /** Set values of angles etc directly without recomputing other parameters. */
  void SetVarAngleScaleShear(
    ScalarArrayType angle,
    ScalarArrayType shear,
    ScalarArrayType scale );

  /** Compute the components of the rotation matrix in the superclass. */
  void ComputeMatrix( void );

  void ComputeMatrixParameters( void );

  /** Update the m_JacobianOfSpatialJacobian.  */
  virtual void PrecomputeJacobianOfSpatialJacobian( void );

private:

  AffineDTI2DTransform( const Self & ); // purposely not implemented
  void operator=( const Self & );       // purposely not implemented

  ScalarArrayType m_Angle;
  ScalarArrayType m_Shear;
  ScalarArrayType m_Scale;

};

}  // namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#include "itkAffineDTI2DTransform.hxx"
#endif

#endif /* __itkAffineDTI2DTransform_h */

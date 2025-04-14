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
  Module:    $RCSfile: itkAdvancedLimitedEuler3DTransform.h,v $
  Language:  C++
  Date:      $Date: 2008-10-13 15:36:31 $
  Version:   $Revision: 1.14 $

  Copyright (c) Insight Software Consortium. All rights reserved.
  See ITKCopyright.txt or http://www.itk.org/HTML/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notices for more information.

=========================================================================*/
#ifndef __itkAdvancedLimitedEuler3DTransform_h
#define __itkAdvancedLimitedEuler3DTransform_h

#include <iostream>
#include "itkAdvancedRigid3DTransform.h"

namespace itk
{

/** \class AdvancedLimitedEuler3DTransform
 *
 * \brief AdvancedLimitedEuler3DTransform of a vector space (e.g. space coordinates)
 *
 * This transform applies a rotation and translation to the space given 3 euler
 * angles and a 3D translation. Rotation is about a user specified center.
 *
 * The parameters for this transform can be set either using individual Set
 * methods or in serialized form using SetParameters() and SetFixedParameters().
 *
 * The serialization of the optimizable parameters is an array of 6 elements.
 * The first 3 represents three euler angle of rotation respectively about
 * the X, Y and Z axis. The last 3 parameters defines the translation in each
 * dimension.
 *
 * The serialization of the fixed parameters is an array of 3 elements defining
 * the center of rotation.
 *
 * \ingroup Transforms
 */
template< class TScalarType = double >
// Data type for scalars (float or double)
class ITK_EXPORT AdvancedLimitedEuler3DTransform :
  public         AdvancedRigid3DTransform< TScalarType >
{
public:

  /** Standard class typedefs. */
  typedef AdvancedLimitedEuler3DTransform                Self;
  typedef AdvancedRigid3DTransform< TScalarType > Superclass;
  typedef SmartPointer< Self >                    Pointer;
  typedef SmartPointer< const Self >              ConstPointer;

  /** New macro for creation of through a Smart Pointer. */
  itkNewMacro( Self );

  /** Run-time type information (and related methods). */
  itkTypeMacro( AdvancedLimitedEuler3DTransform, AdvancedRigid3DTransform );

  /** Dimension of the space. */
  itkStaticConstMacro( SpaceDimension, unsigned int, 3 );
  itkStaticConstMacro( InputSpaceDimension, unsigned int, 3 );
  itkStaticConstMacro( OutputSpaceDimension, unsigned int, 3 );
  itkStaticConstMacro( ParametersDimension, unsigned int, 6 );

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

  /** Set/Get the transformation from a container of parameters
   * This is typically used by optimizers.  There are 6 parameters. The first
   * three represent the angles to rotate around the coordinate axis, and the
   * last three represents the offset. */
  void SetParameters( const ParametersType & parameters ) override;

  const ParametersType & GetParameters( void ) const override;

  /** Set the rotational part of the transform. */
  void SetRotation( ScalarType angleX, ScalarType angleY, ScalarType angleZ );

  itkGetConstMacro( AngleX, ScalarType );
  itkGetConstMacro( AngleY, ScalarType );
  itkGetConstMacro( AngleZ, ScalarType );

  /** Compute the Jacobian of the transformation. */
  void GetJacobian(
    const InputPointType &,
    JacobianType &,
    NonZeroJacobianIndicesType & ) const override;

  /** Set/Get the order of the computation. Default ZXY */
  itkSetMacro( ComputeZYX, bool );
  itkGetConstMacro( ComputeZYX, bool );

  void SetIdentity( void ) override;
  
  /** Set/Get the whether the scales estimation is in progress */
  itkSetMacro(ScalesEstimation, bool);

  /** Set/Get the sharpness aka smoothness of parameter limits */
  itkSetMacro(SharpnessOfLimits, double);
  itkGetConstMacro(SharpnessOfLimits, double);

  /** Update sharpness of limits parameter for each axis based on the inverval width */
  virtual void UpdateSharpnessOfLimitsVector();

  /** Setters/getters for the upper/lower limits */
  virtual void SetUpperLimits(const ParametersType & upperLimits);  
  virtual const ParametersType& GetUpperLimits();

  virtual void SetLowerLimits(const ParametersType & lowerLimits);
  virtual const ParametersType& GetLowerLimits();

  /** Upper/Lower limits reached indicator. Values close to 1 indicate that limit was reached. */
  virtual const ParametersType& GetUpperLimitsReached();
  virtual const ParametersType& GetLowerLimitsReached();

protected:

  AdvancedLimitedEuler3DTransform();
  AdvancedLimitedEuler3DTransform( const MatrixType & matrix,
    const OutputPointType & offset );
  AdvancedLimitedEuler3DTransform( unsigned int paramsSpaceDims );

  ~AdvancedLimitedEuler3DTransform() override {}

  void PrintSelf( std::ostream & os, Indent indent ) const override;

  /** Set values of angles directly without recomputing other parameters. */
  void SetVarRotation( ScalarType angleX, ScalarType angleY, ScalarType angleZ );

  /** Compute the components of the rotation matrix in the superclass. */
  void ComputeMatrix( void ) override;

  void ComputeMatrixParameters( void ) override;

  /** Update the m_JacobianOfSpatialJacobian.  */
  virtual void PrecomputeJacobianOfSpatialJacobian( void );

  /** Initialize limits parameters to default values */
  void InitLimitParameters();

  /** Apply Softplus limiting function to input value and based on range parameters */
  ScalarType SoftplusLimit(const ScalarType & inputValue, const ScalarType & limitValue, const ScalarType & sharpnessOfLimits, bool setHighLimit) const;

  /** Compute derivative of Softplus function at input value and based on range parameters  */
  ScalarType DerivativeSoftplusLimit(const ScalarType & inputValue, const ScalarType & limitValue, const ScalarType & sharpnessOfLimits, bool setHighLimit) const;

private:

  AdvancedLimitedEuler3DTransform( const Self & ); //purposely not implemented
  void operator=( const Self & );           //purposely not implemented

  ScalarType m_AngleX;
  ScalarType m_AngleY;
  ScalarType m_AngleZ;
  bool       m_ComputeZYX;

  ScalarType m_SharpnessOfLimits;
  ParametersType m_SharpnessOfLimitsVector;
  ParametersType m_UpperLimits;
  ParametersType m_LowerLimits;
  ParametersType m_UpperLimitsReached;
  ParametersType m_LowerLimitsReached;
  bool m_ScalesEstimation;
}; //class AdvancedLimitedEuler3DTransform

}  // namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#include "itkAdvancedLimitedEuler3DTransform.hxx"
#endif

#endif /* __itkAdvancedLimitedEuler3DTransform_h */

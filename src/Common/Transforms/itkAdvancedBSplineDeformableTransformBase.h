/*======================================================================

This file is part of the elastix software.

Copyright (c) University Medical Center Utrecht. All rights reserved.
See src/CopyrightElastix.txt or http://elastix.isi.uu.nl/legal.php for
details.

This software is distributed WITHOUT ANY WARRANTY; without even
the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
PURPOSE. See the above copyright notices for more information.

======================================================================*/
#ifndef __itkAdvancedBSplineDeformableTransformBase_h
#define __itkAdvancedBSplineDeformableTransformBase_h

#include "itkAdvancedTransform.h"
#include "itkImage.h"
#include "itkImageRegion.h"

namespace itk
{

/** \class AdvancedBSplineDeformableTransformBase
 * \brief Base class for deformable transform using a B-spline representation
 *
 * This class is the base for the encapsulation of a deformable transform
 * of points from one N-dimensional one space to another N-dimensional space.
 *
 * This class is not templated over the spline order, which makes the use of
 * different spline orders more convenient in subsequent code.
 *
 */
template<
class TScalarType        = double,       // Data type for scalars
unsigned int NDimensions = 3 >
// Number of dimensions
class AdvancedBSplineDeformableTransformBase :
  public AdvancedTransform< TScalarType, NDimensions, NDimensions >
{
public:

  /** Standard class typedefs. */
  typedef AdvancedBSplineDeformableTransformBase Self;
  typedef AdvancedTransform<
    TScalarType, NDimensions, NDimensions >         Superclass;
  typedef SmartPointer< Self >       Pointer;
  typedef SmartPointer< const Self > ConstPointer;

  /** Run-time type information (and related methods). */
  itkTypeMacro( AdvancedBSplineDeformableTransformBase, AdvancedTransform );

  /** Dimension of the domain space. */
  itkStaticConstMacro( SpaceDimension, unsigned int, NDimensions );

  /** Typedefs from Superclass. */
  typedef typename Superclass::ParametersType         ParametersType;
  typedef typename Superclass::ParametersValueType    ParametersValueType;
  typedef typename Superclass::NumberOfParametersType NumberOfParametersType;
  typedef typename Superclass::DerivativeType         DerivativeType;
  typedef typename Superclass::JacobianType           JacobianType;
  typedef typename Superclass::ScalarType             ScalarType;
  typedef typename Superclass::InputPointType         InputPointType;
  typedef typename Superclass::OutputPointType        OutputPointType;
  typedef typename Superclass::InputVectorType        InputVectorType;
  typedef typename Superclass::OutputVectorType       OutputVectorType;
  typedef typename Superclass::InputVnlVectorType     InputVnlVectorType;
  typedef typename Superclass::OutputVnlVectorType    OutputVnlVectorType;
  typedef typename Superclass::InputCovariantVectorType
    InputCovariantVectorType;
  typedef typename Superclass::OutputCovariantVectorType
    OutputCovariantVectorType;
  typedef typename Superclass::TransformCategoryType TransformCategoryType;

  typedef typename Superclass
    ::NonZeroJacobianIndicesType NonZeroJacobianIndicesType;
  typedef typename Superclass::SpatialJacobianType SpatialJacobianType;
  typedef typename Superclass
    ::JacobianOfSpatialJacobianType JacobianOfSpatialJacobianType;
  typedef typename Superclass::SpatialHessianType SpatialHessianType;
  typedef typename Superclass
    ::JacobianOfSpatialHessianType JacobianOfSpatialHessianType;
  typedef typename Superclass::InternalMatrixType           InternalMatrixType;
  typedef typename Superclass::MovingImageGradientType      MovingImageGradientType;
  typedef typename Superclass::MovingImageGradientValueType MovingImageGradientValueType;

  /** This method sets the parameters of the transform.
     * For a B-spline deformation transform, the parameters are the BSpline
     * coefficients on a sparse grid.
     *
     * The parameters are N number of N-D grid of coefficients. Each N-D grid
     * is represented as a flat array of doubles
     * (in the same configuration as an itk::Image).
     * The N arrays are then concatenated to form one parameter array.
     *
     * For efficiency, this transform does not make a copy of the parameters.
     * It only keeps a pointer to the input parameters. It assumes that the memory
     * is managed by the caller. Use SetParametersByValue to force the transform
     * to call copy the parameters.
     *
     * This method wraps each grid as itk::Image's using the user specified
     * grid region, spacing and origin.
     * NOTE: The grid region, spacing and origin must be set first.
     */
  void SetParameters( const ParametersType & parameters );

  /** This method sets the fixed parameters of the transform.
   * For a B-spline deformation transform, the parameters are the following:
   *    Grid Size, Grid Origin, and Grid Spacing
   *
   * The fixed parameters are the three times the size of the templated
   * dimensions.
   * This function has the effect of make the following calls:
   *       transform->SetGridSpacing( spacing );
   *       transform->SetGridOrigin( origin );
   *       transform->SetGridDirection( direction );
   *       transform->SetGridRegion( bsplineRegion );
   *
   * This function was added to allow the transform to work with the
   * itkTransformReader/Writer I/O filters.
   */
  void SetFixedParameters( const ParametersType & parameters );

  /** This method sets the parameters of the transform.
   * For a B-spline deformation transform, the parameters are the BSpline
   * coefficients on a sparse grid.
   *
   * The parameters are N number of N-D grid of coefficients. Each N-D grid
   * is represented as a flat array of doubles
   * (in the same configuration as an itk::Image).
   * The N arrays are then concatenated to form one parameter array.
   *
   * This methods makes a copy of the parameters while for
   * efficiency the SetParameters method does not.
   *
   * This method wraps each grid as itk::Image's using the user specified
   * grid region, spacing and origin.
   * NOTE: The grid region, spacing and origin must be set first.
   */
  void SetParametersByValue( const ParametersType & parameters );

  /** This method can ONLY be invoked AFTER calling SetParameters().
   *  This restriction is due to the fact that the AdvancedBSplineDeformableTransform
   *  does not copy the array of parameters internally, instead it keeps a
   *  pointer to the user-provided array of parameters. This method is also
   *  in violation of the const-correctness of the parameters since the
   *  parameter array has been passed to the transform on a 'const' basis but
   *  the values get modified when the user invokes SetIdentity().
   */
  void SetIdentity( void );

  /** Get the Transformation Parameters. */
  virtual const ParametersType & GetParameters( void ) const;

  /** Get the Transformation Fixed Parameters. */
  virtual const ParametersType & GetFixedParameters( void ) const;

  /** Parameters as SpaceDimension number of images. */
  typedef typename ParametersType::ValueType PixelType;
  typedef Image< PixelType,
    itkGetStaticConstMacro( SpaceDimension ) >           ImageType;
  typedef typename ImageType::Pointer ImagePointer;

  /** Get the array of coefficient images. */
  virtual const ImagePointer * GetCoefficientImages( void ) const
  { return this->m_CoefficientImages; }

  /** Set the array of coefficient images.
   *
   * This is an alternative API for setting the B-spline coefficients
   * as an array of SpaceDimension images. The grid region spacing
   * and origin is taken from the first image. It is assume that
   * the buffered region of all the subsequent images are the same
   * as the first image. Note that no error checking is done.
   *
   * Warning: use either the SetParameters() or SetCoefficientImages()
   * API. Mixing the two modes may results in unexpected results.
   */
  virtual void SetCoefficientImages( ImagePointer images[] );

  /** Typedefs for specifying the extend to the grid. */
  typedef ImageRegion< itkGetStaticConstMacro( SpaceDimension ) > RegionType;

  typedef typename RegionType::IndexType    IndexType;
  typedef typename RegionType::SizeType     SizeType;
  typedef typename ImageType::SpacingType   SpacingType;
  typedef typename ImageType::DirectionType DirectionType;
  typedef typename ImageType::PointType     OriginType;
  typedef IndexType                         GridOffsetType;

  /** This method specifies the region over which the grid resides. */
  virtual void SetGridRegion( const RegionType & region ) = 0;

  //itkGetMacro( GridRegion, RegionType );
  itkGetConstMacro( GridRegion, RegionType );

  /** This method specifies the grid spacing or resolution. */
  virtual void SetGridSpacing( const SpacingType & spacing );

  //itkGetMacro( GridSpacing, SpacingType );
  itkGetConstMacro( GridSpacing, SpacingType );

  /** This method specifies the grid directions . */
  virtual void SetGridDirection( const DirectionType & direction );

  //itkGetMacro( GridDirection, DirectionType );
  itkGetConstMacro( GridDirection, DirectionType );

  /** This method specifies the grid origin. */
  virtual void SetGridOrigin( const OriginType & origin );

  //itkGetMacro( GridOrigin, OriginType );
  itkGetConstMacro( GridOrigin, OriginType );

  /** Parameter index array type. */
  typedef Array< unsigned long > ParameterIndexArrayType;

  /** Method to transform a vector -
   *  not applicable for this type of transform.
   */
  virtual OutputVectorType TransformVector( const InputVectorType & ) const
  {
    itkExceptionMacro( << "Method not applicable for deformable transform." );
    return OutputVectorType();
  }


  /** Method to transform a vnl_vector -
   *  not applicable for this type of transform.
   */
  virtual OutputVnlVectorType TransformVector( const InputVnlVectorType & ) const
  {
    itkExceptionMacro( << "Method not applicable for deformable transform. " );
    return OutputVnlVectorType();
  }


  /** Method to transform a CovariantVector -
   *  not applicable for this type of transform.
   */
  virtual OutputCovariantVectorType TransformCovariantVector(
    const InputCovariantVectorType & ) const
  {
    itkExceptionMacro( << "Method not applicable for deformable transform. " );
    return OutputCovariantVectorType();
  }


  /** Return the number of parameters that completely define the Transform. */
  virtual NumberOfParametersType GetNumberOfParameters( void ) const;

  /** Return the number of parameters per dimension */
  virtual NumberOfParametersType GetNumberOfParametersPerDimension( void ) const;

  /** Return the region of the grid wholly within the support region */
  itkGetConstReferenceMacro( ValidRegion, RegionType );

  /** Indicates that this transform is linear. That is, given two
   * points P and Q, and scalar coefficients a and b, then
   *
   *           T( a*P + b*Q ) = a * T(P) + b * T(Q)
   */
  virtual bool IsLinear( void ) const { return false; }

  /** Indicates the category transform.
   *  e.g. an affine transform, or a local one, e.g. a deformation field.
   */
  virtual TransformCategoryType GetTransformCategory( void ) const
  {
    return Self::BSpline;
  }


  virtual unsigned int GetNumberOfAffectedWeights( void ) const = 0;

  virtual NumberOfParametersType GetNumberOfNonZeroJacobianIndices( void ) const = 0;

  /** This typedef should be equal to the typedef used
   * in derived classes based on the weights function.
   */
  typedef ContinuousIndex< ScalarType, SpaceDimension > ContinuousIndexType;

protected:

  /** Print contents of an AdvancedBSplineDeformableTransformBase. */
  virtual void PrintSelf( std::ostream & os, Indent indent ) const;

  AdvancedBSplineDeformableTransformBase();
  virtual ~AdvancedBSplineDeformableTransformBase();

  /** Wrap flat array into images of coefficients. */
  void WrapAsImages( void );

  /** Convert an input point to a continuous index inside the B-spline grid. */
  void TransformPointToContinuousGridIndex(
    const InputPointType & point, ContinuousIndexType & index ) const;

  void UpdatePointIndexConversions( void );

  virtual void ComputeNonZeroJacobianIndices(
    NonZeroJacobianIndicesType & nonZeroJacobianIndices,
    const RegionType & supportRegion ) const = 0;

  /** Check if a continuous index is inside the valid region. */
  virtual bool InsideValidRegion( const ContinuousIndexType & index ) const;

  /** Array of images representing the B-spline coefficients
   *  in each dimension.
   */
  ImagePointer m_CoefficientImages[ NDimensions ];

  /** Variables defining the coefficient grid extend. */
  RegionType     m_GridRegion;
  SpacingType    m_GridSpacing;
  DirectionType  m_GridDirection;
  OriginType     m_GridOrigin;
  GridOffsetType m_GridOffsetTable;

  DirectionType                                       m_PointToIndexMatrix;
  SpatialJacobianType                                 m_PointToIndexMatrix2;
  DirectionType                                       m_PointToIndexMatrixTransposed;
  SpatialJacobianType                                 m_PointToIndexMatrixTransposed2;
  FixedArray< ScalarType, NDimensions >               m_PointToIndexMatrixDiagonal;
  FixedArray< ScalarType, NDimensions * NDimensions > m_PointToIndexMatrixDiagonalProducts;
  DirectionType                                       m_IndexToPoint;
  bool                                                m_PointToIndexMatrixIsDiagonal;

  RegionType m_ValidRegion;

  /** Variables defining the interpolation support region. */
  unsigned long       m_Offset;
  SizeType            m_SupportSize;
  ContinuousIndexType m_ValidRegionBegin;
  ContinuousIndexType m_ValidRegionEnd;

  /** Odd or even order B-spline. */
  bool m_SplineOrderOdd;

  /** Keep a pointer to the input parameters. */
  const ParametersType * m_InputParametersPointer;

  /** Jacobian as SpaceDimension number of images. */
  typedef typename JacobianType::ValueType JacobianPixelType;
  typedef Image< JacobianPixelType,
    itkGetStaticConstMacro( SpaceDimension ) >  JacobianImageType;

  typename JacobianImageType::Pointer m_JacobianImage[ NDimensions ];

  /** Keep track of last support region used in computing the Jacobian
  * for fast resetting of Jacobian to zero.
  */
  mutable IndexType m_LastJacobianIndex;

  /** Array holding images wrapped from the flat parameters. */
  ImagePointer m_WrappedImage[ NDimensions ];

  /** Internal parameters buffer. */
  ParametersType m_InternalParametersBuffer;

  void UpdateGridOffsetTable( void );

private:

  AdvancedBSplineDeformableTransformBase( const Self & ); // purposely not implemented
  void operator=( const Self & );                         // purposely not implemented

};

}  // namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#include "itkAdvancedBSplineDeformableTransformBase.hxx"
#endif

#endif /* __itkAdvancedBSplineDeformableTransformBase_h */

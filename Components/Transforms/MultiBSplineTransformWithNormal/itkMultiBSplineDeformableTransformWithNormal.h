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
#ifndef itkMultiBSplineDeformableTransformWithNormal_h
#define itkMultiBSplineDeformableTransformWithNormal_h

#include "itkAdvancedBSplineDeformableTransform.h"
#include "itkNearestNeighborInterpolateImageFunction.h"

namespace itk
{
/**
 * \class MultiBSplineDeformableTransformWithNormal
 * \brief This transform is a composition of B-spline transformations,
 *   allowing sliding motion between different labels.
 *
 * Detailed explanation ...
 *
 * \author Vivien Delmon
 *
 * \ingroup Transforms
 */

template <class TScalarType = double,   // Data type for scalars
          unsigned int NDimensions = 3, // Number of dimensions
          unsigned int VSplineOrder = 3>
// Spline order
class ITK_TEMPLATE_EXPORT MultiBSplineDeformableTransformWithNormal
  : public AdvancedTransform<TScalarType, NDimensions, NDimensions>
{
public:
  ITK_DISALLOW_COPY_AND_MOVE(MultiBSplineDeformableTransformWithNormal);

  /** Standard class typedefs. */
  using Self = MultiBSplineDeformableTransformWithNormal;
  using Superclass = AdvancedTransform<TScalarType, NDimensions, NDimensions>;
  using Pointer = SmartPointer<Self>;
  using ConstPointer = SmartPointer<const Self>;

  /** New macro for creation of through the object factory. */
  itkNewMacro(Self);

  /** Run-time type information (and related methods). */
  itkTypeMacro(MultiBSplineDeformableTransformWithNormal, AdvancedTransform);

  /** Dimension of the domain space. */
  itkStaticConstMacro(SpaceDimension, unsigned int, NDimensions);

  /** The BSpline order. */
  itkStaticConstMacro(SplineOrder, unsigned int, VSplineOrder);

  /** Typedefs from Superclass. */
  using typename Superclass::ParametersType;
  using typename Superclass::NumberOfParametersType;
  using typename Superclass::JacobianType;
  using typename Superclass::ScalarType;
  using typename Superclass::InputPointType;
  using typename Superclass::OutputPointType;
  using typename Superclass::InputVectorType;
  using typename Superclass::OutputVectorType;
  using typename Superclass::InputVnlVectorType;
  using typename Superclass::OutputVnlVectorType;
  using typename Superclass::InputCovariantVectorType;
  using typename Superclass::OutputCovariantVectorType;

  using typename Superclass::NonZeroJacobianIndicesType;
  using typename Superclass::SpatialJacobianType;
  using typename Superclass::JacobianOfSpatialJacobianType;
  using typename Superclass::SpatialHessianType;
  using typename Superclass::JacobianOfSpatialHessianType;
  using typename Superclass::InternalMatrixType;

  /** Interpolation weights function type. */
  using WeightsFunctionType = BSplineInterpolationWeightFunction2<ScalarType, Self::SpaceDimension, VSplineOrder>;
  using WeightsType = typename WeightsFunctionType::WeightsType;

  /** This method sets the parameters of the transform.
   * For a BSpline deformation transform, the parameters are the BSpline
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
  void
  SetParameters(const ParametersType & parameters) override;

  /** This method sets the fixed parameters of the transform.
   * For a BSpline deformation transform, the parameters are the following:
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
  void
  SetFixedParameters(const ParametersType & parameters) override;

  /** This method sets the parameters of the transform.
   * For a BSpline deformation transform, the parameters are the BSpline
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
  void
  SetParametersByValue(const ParametersType & parameters) override;

  /** This method can ONLY be invoked AFTER calling SetParameters().
   *  This restriction is due to the fact that the AdvancedBSplineDeformableTransform
   *  does not copy the array of parameters internally, instead it keeps a
   *  pointer to the user-provided array of parameters. This method is also
   *  in violation of the const-correctness of the parameters since the
   *  parameter array has been passed to the transform on a 'const' basis but
   *  the values get modified when the user invokes SetIdentity().
   */
  void
  SetIdentity();

  /** Get the Transformation Parameters. */
  const ParametersType &
  GetParameters() const override;

  /** Get the Transformation Fixed Parameters. */
  const ParametersType &
  GetFixedParameters() const override;

  /** Parameters as SpaceDimension number of images. */
  using PixelType = typename ParametersType::ValueType;
  using ImageType = Image<PixelType, Self::SpaceDimension>;
  using ImagePointer = typename ImageType::Pointer;

  /** Get the array of coefficient images. */
  // virtual ImagePointer * GetCoefficientImage()
  //  { return this->m_CoefficientImage; }
  // virtual const ImagePointer * GetCoefficientImage() const
  //  { return this->m_CoefficientImage; }

  /** Set the array of coefficient images.
   *
   * This is an alternative API for setting the BSpline coefficients
   * as an array of SpaceDimension images. The grid region spacing
   * and origin is taken from the first image. It is assume that
   * the buffered region of all the subsequent images are the same
   * as the first image. Note that no error checking is done.
   *
   * Warning: use either the SetParameters() or SetCoefficientImages()
   * API. Mixing the two modes may results in unexpected results.
   */
  // virtual void SetCoefficientImage( ImagePointer images[] );

  /** Typedefs for specifying the extend to the grid. */
  using RegionType = ImageRegion<Self::SpaceDimension>;

  using IndexType = typename RegionType::IndexType;
  using SizeType = typename RegionType::SizeType;
  using SpacingType = typename ImageType::SpacingType;
  using DirectionType = typename ImageType::DirectionType;
  using OriginType = typename ImageType::PointType;
  using GridOffsetType = IndexType;

  /** This method specifies the region over which the grid resides. */
  virtual void
  SetGridRegion(const RegionType & region);

  virtual RegionType
  GetGridRegion() const;

  /** This method specifies the grid spacing or resolution. */
  virtual void
  SetGridSpacing(const SpacingType & spacing);

  virtual SpacingType
  GetGridSpacing() const;

  /** This method specifies the grid directions . */
  virtual void
  SetGridDirection(const DirectionType & spacing);

  virtual DirectionType
  GetGridDirection() const;

  /** This method specifies the grid origin. */
  virtual void
  SetGridOrigin(const OriginType & origin);

  virtual OriginType
  GetGridOrigin() const;

  /** Typedef of the label image. */
  using ImageLabelType = Image<unsigned char, Self::SpaceDimension>;
  using ImageLabelPointer = typename ImageLabelType::Pointer;

  using ImageLabelInterpolator = itk::NearestNeighborInterpolateImageFunction<ImageLabelType, TScalarType>;
  using ImageLabelInterpolatorPointer = typename ImageLabelInterpolator::Pointer;

  /** Typedef of the Normal Grid. */
  using VectorType = Vector<TScalarType, Self::SpaceDimension>;
  using BaseType = Vector<VectorType, Self::SpaceDimension>;
  using ImageVectorType = Image<VectorType, Self::SpaceDimension>;
  using ImageVectorPointer = typename ImageVectorType::Pointer;
  using ImageBaseType = Image<BaseType, Self::SpaceDimension>;
  using ImageBasePointer = typename ImageBaseType::Pointer;

  /** This method specifies the label image. */
  void
  SetLabels(ImageLabelType * labels);

  itkGetConstMacro(Labels, ImageLabelType *);

  itkGetConstMacro(NbLabels, unsigned char);

  /** Update Local Bases : call to it should become automatic and the function should become private */
  void
  UpdateLocalBases();

  itkGetConstMacro(LocalBases, ImageBaseType *);

  /** Parameter index array type. */
  using ParameterIndexArrayType = Array<unsigned long>;

  /** Method to transform a vector -
   *  not applicable for this type of transform.
   */
  OutputVectorType
  TransformVector(const InputVectorType &) const override
  {
    itkExceptionMacro(<< "Method not applicable for deformable transform.");
    return OutputVectorType();
  }


  /** Method to transform a vnl_vector -
   *  not applicable for this type of transform.
   */
  OutputVnlVectorType
  TransformVector(const InputVnlVectorType &) const override
  {
    itkExceptionMacro(<< "Method not applicable for deformable transform. ");
    return OutputVnlVectorType();
  }


  /** Method to transform a CovariantVector -
   *  not applicable for this type of transform.
   */
  OutputCovariantVectorType
  TransformCovariantVector(const InputCovariantVectorType &) const override
  {
    itkExceptionMacro(<< "Method not applicable for deformable transform. ");
    return OutputCovariantVectorType();
  }


  /** Return the number of parameters that completely define the Transform. */
  NumberOfParametersType
  GetNumberOfParameters() const override;

  /** Return the number of parameters per dimension */
  virtual NumberOfParametersType
  GetNumberOfParametersPerDimension() const;

  /** Return the region of the grid wholly within the support region */
  virtual const RegionType &
  GetValidRegion()
  {
    return m_Trans[0]->GetValidRegion();
  }


  /** Indicates that this transform is linear. That is, given two
   * points P and Q, and scalar coefficients a and b, then
   *
   *           T( a*P + b*Q ) = a * T(P) + b * T(Q)
   */
  bool
  IsLinear() const override
  {
    return false;
  }

  /** Get number of weights. */
  virtual unsigned long
  GetNumberOfWeights() const
  {
    return m_Trans[0]->m_WeightsFunction->GetNumberOfWeights();
  }


  virtual unsigned int
  GetNumberOfAffectedWeights() const
  {
    return m_Trans[0]->m_WeightsFunction->GetNumberOfWeights();
  }


  NumberOfParametersType
  GetNumberOfNonZeroJacobianIndices() const override
  {
    return m_Trans[0]->m_WeightsFunction->GetNumberOfWeights() * SpaceDimension;
  }


  /** Whether the advanced transform has nonzero matrices. */
  virtual bool
  GetHasNonZeroSpatialJacobian() const
  {
    return true;
  }


  virtual bool
  HasNonZeroJacobianOfSpatialJacobian() const
  {
    return true;
  }


  bool
  GetHasNonZeroSpatialHessian() const override
  {
    return true;
  }


  virtual bool
  HasNonZeroJacobianOfSpatialHessian() const
  {
    return true;
  }


  /** This typedef should be equal to the typedef used in derived classes based on the weightsfunction. */
  using ContinuousIndexType = ContinuousIndex<ScalarType, SpaceDimension>;

  /** Transform points by a BSpline deformable transformation. */
  OutputPointType
  TransformPoint(const InputPointType & point) const override;

  /** Compute the Jacobian matrix of the transformation at one point. */
  // virtual const JacobianType & GetJacobian( const InputPointType & point ) const;

  /** Compute the Jacobian of the transformation. */
  void
  GetJacobian(const InputPointType & inputPoint, JacobianType & j, NonZeroJacobianIndicesType &) const override;

  /** Compute the spatial Jacobian of the transformation. */
  void
  GetSpatialJacobian(const InputPointType & inputPoint, SpatialJacobianType & sj) const override;

  void
  GetJacobianOfSpatialJacobian(const InputPointType &          inputPoint,
                               JacobianOfSpatialJacobianType & jsj,
                               NonZeroJacobianIndicesType &    nonZeroJacobianIndices) const override;

  void
  GetJacobianOfSpatialJacobian(const InputPointType &,
                               SpatialJacobianType &,
                               JacobianOfSpatialJacobianType &,
                               NonZeroJacobianIndicesType &) const override;

  /** Compute the spatial Hessian of the transformation. */
  void
  GetSpatialHessian(const InputPointType & inputPoint, SpatialHessianType & sh) const override;

  void
  GetJacobianOfSpatialHessian(const InputPointType &         inputPoint,
                              JacobianOfSpatialHessianType & jsh,
                              NonZeroJacobianIndicesType &   nonZeroJacobianIndices) const override
  {
    itkExceptionMacro(<< "ERROR: GetJacobianOfSpatialHessian() not yet implemented in the "
                         "MultiBSplineDeformableTransformWithNormal class.");
  }


  void
  GetJacobianOfSpatialHessian(const InputPointType &,
                              SpatialHessianType &,
                              JacobianOfSpatialHessianType &,
                              NonZeroJacobianIndicesType &) const override;

protected:
  /** Print contents of an MultiBSplineDeformableTransformWithNormal. */
  void
  PrintSelf(std::ostream & os, Indent indent) const override;

  MultiBSplineDeformableTransformWithNormal();
  ~MultiBSplineDeformableTransformWithNormal() override = default;

  /** Wrap flat array into images of coefficients. */
  // void WrapAsImages();

  /** Convert an input point to a continuous index inside the BSpline grid. */
  /*
  void TransformPointToContinuousGridIndex(
   const InputPointType & point, ContinuousIndexType & index ) const;

  virtual void ComputeNonZeroJacobianIndices(
    NonZeroJacobianIndicesType & nonZeroJacobianIndices,
    const RegionType & supportRegion ) const;
    */

  /** Check if a continuous index is inside the valid region. */
  // virtual bool InsideValidRegion( const ContinuousIndexType& index ) const;

  /** The bulk transform. */
  // BulkTransformPointer  m_BulkTransform;

  /** Array of images representing the B-spline coefficients
   *  in each dimension. */
  // ImagePointer    m_CoefficientImage[ NDimensions ];

  /** Variables defining the coefficient grid extend. */
  // RegionType          m_GridRegion;
  // SpacingType         m_GridSpacing;
  // DirectionType       m_GridDirection;
  // OriginType          m_GridOrigin;
  // GridOffsetType      m_GridOffsetTable;

  // DirectionType       m_PointToIndexMatrix;
  // SpatialJacobianType m_PointToIndexMatrix2;
  // DirectionType       m_PointToIndexMatrixTransposed;
  // SpatialJacobianType m_PointToIndexMatrixTransposed2;
  // DirectionType       m_IndexToPoint;

  // RegionType      m_ValidRegion;

  /** Variables defining the interpolation support region. */
  // unsigned long   m_Offset;
  // SizeType        m_SupportSize;
  // ContinuousIndexType m_ValidRegionBegin;
  // ContinuousIndexType m_ValidRegionEnd;

  /** Odd or even order BSpline. */
  // bool m_SplineOrderOdd;

  /** Keep a pointer to the input parameters. */
  const ParametersType * m_InputParametersPointer;

  /** Jacobian as SpaceDimension number of images. */
  /*
  using JacobianPixelType = typename JacobianType::ValueType;
  using JacobianImageType = Image< JacobianPixelType,
    itkGetStaticConstMacro( SpaceDimension ) >;

  typename JacobianImageType::Pointer m_JacobianImage[ NDimensions ];
  */

  /** Keep track of last support region used in computing the Jacobian
   * for fast resetting of Jacobian to zero.
   */
  // mutable IndexType m_LastJacobianIndex;

  /** Array holding images wrapped from the flat parameters. */
  // ImagePointer    m_WrappedImage[ NDimensions ];

  /** Internal parameters buffer. */
  ParametersType m_InternalParametersBuffer;

  using TransformType = AdvancedBSplineDeformableTransform<TScalarType, Self::SpaceDimension, VSplineOrder>;

  unsigned char                                m_NbLabels;
  ImageLabelPointer                            m_Labels;
  ImageLabelInterpolatorPointer                m_LabelsInterpolator;
  ImageVectorPointer                           m_LabelsNormals;
  std::vector<typename TransformType::Pointer> m_Trans;
  std::vector<ParametersType>                  m_Para;
  mutable int                                  m_LastJacobian;
  ImageBasePointer                             m_LocalBases;

private:
  void
  DispatchParameters(const ParametersType & parameters);

  void
  PointToLabel(const InputPointType & p, int & l) const;
};

} // end namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#  include "itkMultiBSplineDeformableTransformWithNormal.hxx"
#endif

#endif // end itkMultiBSplineDeformableTransformWithNormal_h

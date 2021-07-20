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
#ifndef itkPolydataDummyPenalty_h
#define itkPolydataDummyPenalty_h

#include "itkSingleValuedPointSetToPointSetMetric.h"
#include "itkPoint.h"
#include "itkPointSet.h"
#include "itkImage.h"
#include "itkMesh.h"
#include <itkVectorContainer.h>

namespace itk
{

/** \class MeshPenalty
 * \brief A dummy metric to generate transformed meshes each iteration.
 *
 *
 *
 * \ingroup RegistrationMetrics
 */

template <class TFixedPointSet, class TMovingPointSet>
class ITK_TEMPLATE_EXPORT MeshPenalty : public SingleValuedPointSetToPointSetMetric<TFixedPointSet, TMovingPointSet>
{
public:
  /** Standard class typedefs. */
  typedef MeshPenalty                                                           Self;
  typedef SingleValuedPointSetToPointSetMetric<TFixedPointSet, TMovingPointSet> Superclass;
  typedef SmartPointer<Self>                                                    Pointer;
  typedef SmartPointer<const Self>                                              ConstPointer;

  /** Type used for representing point components  */

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Run-time type information (and related methods). */
  itkTypeMacro(MeshPenalty, SingleValuedPointSetToPointSetMetric);

  /** Types transferred from the base class */
  typedef typename Superclass::TransformType           TransformType;
  typedef typename Superclass::TransformPointer        TransformPointer;
  typedef typename Superclass::TransformParametersType TransformParametersType;
  typedef typename Superclass::TransformJacobianType   TransformJacobianType;

  typedef typename Superclass::MeasureType         MeasureType;
  typedef typename Superclass::DerivativeType      DerivativeType;
  typedef typename Superclass::DerivativeValueType DerivativeValueType;

  /** Typedefs. */
  typedef typename Superclass::InputPointType         InputPointType;
  typedef typename Superclass::OutputPointType        OutputPointType;
  typedef typename InputPointType::CoordRepType       CoordRepType;
  typedef vnl_vector<CoordRepType>                    VnlVectorType;
  typedef typename TransformType::InputPointType      FixedImagePointType;
  typedef typename TransformType::OutputPointType     MovingImagePointType;
  typedef typename TransformType::SpatialJacobianType SpatialJacobianType;

  typedef typename Superclass::NonZeroJacobianIndicesType NonZeroJacobianIndicesType;

  /** Constants for the pointset dimensions. */
  itkStaticConstMacro(FixedPointSetDimension, unsigned int, Superclass::FixedPointSetDimension);

  typedef Vector<typename TransformType::ScalarType, FixedPointSetDimension> PointNormalType;
  typedef unsigned char                                                      DummyMeshPixelType;
  typedef DefaultStaticMeshTraits<PointNormalType, FixedPointSetDimension, FixedPointSetDimension, CoordRepType>
                                                                        MeshTraitsType;
  typedef Mesh<PointNormalType, FixedPointSetDimension, MeshTraitsType> FixedMeshType;

  typedef typename FixedMeshType::ConstPointer FixedMeshConstPointer;
  typedef typename FixedMeshType::Pointer      FixedMeshPointer;
  typedef typename MeshTraitsType::CellType    CellInterfaceType;

  typedef typename FixedMeshType::PointType             MeshPointType;
  typedef typename FixedMeshType::PointType::VectorType VectorType;

  typedef typename FixedMeshType::PointsContainer              MeshPointsContainerType;
  typedef typename MeshPointsContainerType::Pointer            MeshPointsContainerPointer;
  typedef typename MeshPointsContainerType::ConstPointer       MeshPointsContainerConstPointer;
  typedef typename FixedMeshType::PointsContainerConstIterator MeshPointsContainerConstIteratorType;
  typedef typename FixedMeshType::PointsContainerIterator      MeshPointsContainerIteratorType;

  typedef typename FixedMeshType::PointDataContainer             MeshPointDataContainerType;
  typedef typename FixedMeshType::PointDataContainerConstPointer MeshPointDataContainerConstPointer;
  typedef typename FixedMeshType::PointDataContainerPointer      MeshPointDataContainerPointer;
  // typedef typename FixedMeshType::PointDataContainerConstIterator     MeshPointDataContainerConstIteratorType;
  typedef typename FixedMeshType::PointDataContainerIterator MeshPointDataContainerConstIteratorType;
  typedef typename MeshPointDataContainerType::Iterator      MeshPointDataContainerIteratorType;

  typedef unsigned int                                       MeshIdType;
  typedef VectorContainer<MeshIdType, FixedMeshConstPointer> FixedMeshContainerType;
  typedef typename FixedMeshContainerType::Pointer           FixedMeshContainerPointer;
  typedef typename FixedMeshContainerType::ConstPointer      FixedMeshContainerConstPointer;
  typedef typename FixedMeshContainerType::ElementIdentifier FixedMeshContainerElementIdentifier;

  typedef VectorContainer<MeshIdType, FixedMeshPointer>  MappedMeshContainerType;
  typedef typename MappedMeshContainerType::Pointer      MappedMeshContainerPointer;
  typedef typename MappedMeshContainerType::ConstPointer MappedMeshContainerConstPointer;

  typedef Array<DerivativeValueType> MeshPointsDerivativeValueType;

  itkSetConstObjectMacro(FixedMeshContainer, FixedMeshContainerType);
  itkGetConstObjectMacro(FixedMeshContainer, FixedMeshContainerType);

  itkSetObjectMacro(MappedMeshContainer, MappedMeshContainerType);
  itkGetModifiableObjectMacro(MappedMeshContainer, MappedMeshContainerType);

  /** Get the mapped points. */
  // itkGetObjectMacro( MappedPoints, MeshPointsContainerPointer );

  /** Connect the fixed pointset.  */
  // itkSetConstObjectMacro( FixedMesh, FixedMeshType );

  /** Get the fixed pointset. */
  // itkGetConstObjectMacro( FixedMesh, FixedMeshType );

  /** Connect the Transform. */
  // itkSetObjectMacro( Transform, TransformType );

  /** Get a pointer to the Transform.  */
  // itkGetConstObjectMacro( Transform, TransformType );

  /** Set the parameters defining the Transform. */
  // void SetTransformParameters( const ParametersType & parameters ) const;

  /** Return the number of parameters required by the transform. */
  // unsigned int GetNumberOfParameters( void ) const
  //{ return this->m_Transform->GetNumberOfParameters(); }

  /** Initialize the Metric by making sure that all the components are
   *  present and plugged together correctly.
   */
  void
  Initialize(void) override;

  /** Set the fixed mask. */
  // \todo: currently not used
  // itkSetConstObjectMacro( FixedImageMask, FixedImageMaskType );

  /** Get the fixed mask. */
  // itkGetConstObjectMacro( FixedImageMask, FixedImageMaskType );

  /**  Get the value for single valued optimizers. */
  MeasureType
  GetValue(const TransformParametersType & parameters) const override;

  /** Get the derivatives of the match measure. */
  void
  GetDerivative(const TransformParametersType & parameters, DerivativeType & Derivative) const override;

  /**  Get value and derivatives for multiple valued optimizers. */
  void
  GetValueAndDerivative(const TransformParametersType & parameters,
                        MeasureType &                   Value,
                        DerivativeType &                Derivative) const override;

protected:
  MeshPenalty();
  ~MeshPenalty() override;

  /** PrintSelf. */
  void
  PrintSelf(std::ostream & os, Indent indent) const override;

  /** Member variables. */
  mutable FixedMeshContainerConstPointer m_FixedMeshContainer;
  mutable MappedMeshContainerPointer     m_MappedMeshContainer;

private:
  MeshPenalty(const Self &) = delete;
  void
  operator=(const Self &) = delete;
};

} // end namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#  include "itkPolydataDummyPenalty.hxx"
#endif

#endif

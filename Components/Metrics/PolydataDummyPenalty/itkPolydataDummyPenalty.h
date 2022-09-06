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
  ITK_DISALLOW_COPY_AND_MOVE(MeshPenalty);

  /** Standard class typedefs. */
  using Self = MeshPenalty;
  using Superclass = SingleValuedPointSetToPointSetMetric<TFixedPointSet, TMovingPointSet>;
  using Pointer = SmartPointer<Self>;
  using ConstPointer = SmartPointer<const Self>;

  /** Type used for representing point components  */

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Run-time type information (and related methods). */
  itkTypeMacro(MeshPenalty, SingleValuedPointSetToPointSetMetric);

  /** Types transferred from the base class */
  using typename Superclass::TransformType;
  using typename Superclass::TransformPointer;
  using typename Superclass::TransformParametersType;
  using typename Superclass::TransformJacobianType;

  using typename Superclass::MeasureType;
  using typename Superclass::DerivativeType;
  using typename Superclass::DerivativeValueType;

  /** Typedefs. */
  using typename Superclass::InputPointType;
  using typename Superclass::OutputPointType;
  using CoordRepType = typename InputPointType::CoordRepType;
  using VnlVectorType = vnl_vector<CoordRepType>;
  using FixedImagePointType = typename TransformType::InputPointType;
  using MovingImagePointType = typename TransformType::OutputPointType;
  using SpatialJacobianType = typename TransformType::SpatialJacobianType;

  using typename Superclass::NonZeroJacobianIndicesType;

  /** Constants for the pointset dimensions. */
  itkStaticConstMacro(FixedPointSetDimension, unsigned int, Superclass::FixedPointSetDimension);

  using PointNormalType = Vector<typename TransformType::ScalarType, FixedPointSetDimension>;
  using DummyMeshPixelType = unsigned char;
  using MeshTraitsType =
    DefaultStaticMeshTraits<PointNormalType, FixedPointSetDimension, FixedPointSetDimension, CoordRepType>;
  using FixedMeshType = Mesh<PointNormalType, FixedPointSetDimension, MeshTraitsType>;

  using FixedMeshConstPointer = typename FixedMeshType::ConstPointer;
  using FixedMeshPointer = typename FixedMeshType::Pointer;
  using CellInterfaceType = typename MeshTraitsType::CellType;

  using MeshPointType = typename FixedMeshType::PointType;
  using VectorType = typename FixedMeshType::PointType::VectorType;

  using MeshPointsContainerType = typename FixedMeshType::PointsContainer;
  using MeshPointsContainerPointer = typename MeshPointsContainerType::Pointer;
  using MeshPointsContainerConstPointer = typename MeshPointsContainerType::ConstPointer;
  using MeshPointsContainerConstIteratorType = typename FixedMeshType::PointsContainerConstIterator;
  using MeshPointsContainerIteratorType = typename FixedMeshType::PointsContainerIterator;

  using MeshPointDataContainerType = typename FixedMeshType::PointDataContainer;
  using MeshPointDataContainerConstPointer = typename FixedMeshType::PointDataContainerConstPointer;
  using MeshPointDataContainerPointer = typename FixedMeshType::PointDataContainerPointer;
  // typedef typename FixedMeshType::PointDataContainerConstIterator     MeshPointDataContainerConstIteratorType;
  using MeshPointDataContainerConstIteratorType = typename FixedMeshType::PointDataContainerIterator;
  using MeshPointDataContainerIteratorType = typename MeshPointDataContainerType::Iterator;

  using MeshIdType = unsigned int;
  using FixedMeshContainerType = VectorContainer<MeshIdType, FixedMeshConstPointer>;
  using FixedMeshContainerPointer = typename FixedMeshContainerType::Pointer;
  using FixedMeshContainerConstPointer = typename FixedMeshContainerType::ConstPointer;
  using FixedMeshContainerElementIdentifier = typename FixedMeshContainerType::ElementIdentifier;

  using MappedMeshContainerType = VectorContainer<MeshIdType, FixedMeshPointer>;
  using MappedMeshContainerPointer = typename MappedMeshContainerType::Pointer;
  using MappedMeshContainerConstPointer = typename MappedMeshContainerType::ConstPointer;

  using MeshPointsDerivativeValueType = Array<DerivativeValueType>;

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
  // unsigned int GetNumberOfParameters() const
  //{ return this->m_Transform->GetNumberOfParameters(); }

  /** Initialize the Metric by making sure that all the components are
   *  present and plugged together correctly.
   */
  void
  Initialize() override;

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
  ~MeshPenalty() override = default;

  /** PrintSelf. */
  void
  PrintSelf(std::ostream & os, Indent indent) const override;

  /** Member variables. */
  mutable FixedMeshContainerConstPointer m_FixedMeshContainer;
  mutable MappedMeshContainerPointer     m_MappedMeshContainer;
};

} // end namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#  include "itkPolydataDummyPenalty.hxx"
#endif

#endif

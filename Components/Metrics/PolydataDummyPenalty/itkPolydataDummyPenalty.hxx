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
#ifndef itkPolydataDummyPenalty_hxx
#define itkPolydataDummyPenalty_hxx

#include "itkPolydataDummyPenalty.h"

namespace itk
{

/**
 * ******************* Constructor *******************
 */

template <class TFixedPointSet, class TMovingPointSet>
MeshPenalty<TFixedPointSet, TMovingPointSet>::MeshPenalty()
{
  this->m_MappedMeshContainer = MappedMeshContainerType::New();
} // end Constructor


/**
 * *********************** Initialize *****************************
 */

template <class TFixedPointSet, class TMovingPointSet>
void
MeshPenalty<TFixedPointSet, TMovingPointSet>::Initialize()
{
  /** Call the initialize of the superclass. */
  // this->Superclass::Initialize();

  if (!this->m_Transform)
  {
    itkExceptionMacro(<< "Transform is not present");
  }

  if (!this->m_FixedMeshContainer)
  {
    itkExceptionMacro(<< "FixedMeshContainer is not present");
  }

  const FixedMeshContainerElementIdentifier numberOfMeshes = this->m_FixedMeshContainer->Size();
  this->m_MappedMeshContainer->Reserve(numberOfMeshes);

  for (FixedMeshContainerElementIdentifier meshId = 0; meshId < numberOfMeshes; ++meshId)
  {
    FixedMeshConstPointer fixedMesh = this->m_FixedMeshContainer->ElementAt(meshId);
    // If the mesh is provided by a source, update the source.
    if (fixedMesh->GetSource())
    {
      fixedMesh->GetSource()->Update();
    }

    MeshPointsContainerConstPointer fixedPoints = fixedMesh->GetPoints();
    const unsigned int              numberOfPoints = fixedPoints->Size();
    // MeshPointDataContainerConstPointer fixedNormals =  fixedMesh->GetPointData();

    // const unsigned int numberOfNormals = fixedNormals->Size();
    // if ( numberOfPoints!=numberOfNormals )
    //{
    //  itkExceptionMacro( << "numberOfPoints does not match numberOfNormals" );
    //}

    auto mappedPoints = MeshPointsContainerType::New();
    mappedPoints->Reserve(numberOfPoints);

    // auto mappedPointNormals = MeshPointDataContainerType::New();
    // mappedPointNormals->Reserve(numberOfNormals);

    auto mappedMesh = FixedMeshType::New();
    mappedMesh->SetPoints(mappedPoints);

    mappedMesh->SetPointData(nullptr);
    // mappedMesh->SetPointData(mappedPointNormals);

    // mappedMesh was constructed with a Cellscontainer and CellDatacontainer of size 0.
    // We use a null pointer to set them to undefined, which is also the default behavior of the MeshReader.
    // "Write result mesh" checks the null pointer and writes a mesh with the remaining data filled in from the fixed
    // mesh.
    mappedMesh->SetCells(nullptr);
    mappedMesh->SetCellData(nullptr);

    this->m_MappedMeshContainer->SetElement(meshId, mappedMesh);
  }
} // end Initialize()


/**
 * ******************* GetValue *******************
 */

template <class TFixedPointSet, class TMovingPointSet>
auto
MeshPenalty<TFixedPointSet, TMovingPointSet>::GetValue(const TransformParametersType & parameters) const -> MeasureType
{
  /** Sanity checks. */
  FixedMeshContainerConstPointer fixedMeshContainer = this->GetFixedMeshContainer();
  if (!fixedMeshContainer)
  {
    itkExceptionMacro(<< "FixedMeshContainer mesh has not been assigned");
  }

  /** Initialize some variables */
  // this->m_NumberOfPointsCounted = 0;
  MeasureType value = NumericTraits<MeasureType>::Zero;

  // InputPointType movingPoint;
  // OutputPointType fixedPoint;
  /** Get the current corresponding points. */

  /** Make sure the transform parameters are up to date. */
  this->SetTransformParameters(parameters);

  DerivativeType dummyDerivative;
  // TODO: copy paste the necessary parts for the calculation of the value only.
  this->GetValueAndDerivative(parameters, value, dummyDerivative);

  return value;

} // end GetValue()


/**
 * ******************* GetDerivative *******************
 */

template <class TFixedPointSet, class TMovingPointSet>
void
MeshPenalty<TFixedPointSet, TMovingPointSet>::GetDerivative(const TransformParametersType & parameters,
                                                            DerivativeType &                derivative) const
{
  /** When the derivative is calculated, all information for calculating
   * the metric value is available. It does not cost anything to calculate
   * the metric value now. Therefore, we have chosen to only implement the
   * GetValueAndDerivative(), supplying it with a dummy value variable.
   */
  MeasureType dummyvalue = NumericTraits<MeasureType>::Zero;
  this->GetValueAndDerivative(parameters, dummyvalue, derivative);

} // end GetDerivative()


/**
 * ******************* GetValueAndDerivative *******************
 */

template <class TFixedPointSet, class TMovingPointSet>
void
MeshPenalty<TFixedPointSet, TMovingPointSet>::GetValueAndDerivative(const TransformParametersType & parameters,
                                                                    MeasureType &                   value,
                                                                    DerivativeType &                derivative) const
{

  /** Sanity checks. */
  FixedMeshContainerConstPointer fixedMeshContainer = this->GetFixedMeshContainer();
  if (!fixedMeshContainer)
  {
    itkExceptionMacro(<< "FixedMeshContainer mesh has not been assigned");
  }

  /** Initialize some variables */
  value = NumericTraits<MeasureType>::Zero;

  /** Make sure the transform parameters are up to date. */
  this->SetTransformParameters(parameters);

  derivative = DerivativeType(this->GetNumberOfParameters());
  derivative.Fill(NumericTraits<DerivativeValueType>::ZeroValue());

  // NonZeroJacobianIndicesType nzji( this->m_Transform->GetNumberOfNonZeroJacobianIndices() );
  // TransformJacobianType      jacobian;

  const FixedMeshContainerElementIdentifier numberOfMeshes = this->m_FixedMeshContainer->Size();

  /* Loop over all meshes in this Metric*/
  for (FixedMeshContainerElementIdentifier meshId = 0; meshId < numberOfMeshes; ++meshId)
  {
    const FixedMeshConstPointer           fixedMesh = fixedMeshContainer->ElementAt(meshId);
    const MeshPointsContainerConstPointer fixedPoints = fixedMesh->GetPoints();
    // const MeshPointDataContainerConstPointer fixedNormals =  fixedMesh->GetPointData();
    // const unsigned int numberOfPoints = fixedPoints->Size();

    const FixedMeshPointer           mappedMesh = this->m_MappedMeshContainer->ElementAt(meshId);
    const MeshPointsContainerPointer mappedPoints = mappedMesh->GetPoints();
    // const MeshPointDataContainerPointer mappedNormals =  mappedMesh->GetPointData();

    // FixedMeshType::PointsContainer::Pointer derivPoints = FixedMeshType::PointsContainer::New();
    // derivPoints->resize(numberOfPoints);

    MeshPointsContainerConstIteratorType fixedPointIt = fixedPoints->Begin();
    MeshPointsContainerIteratorType      mappedPointIt = mappedPoints->Begin();
    MeshPointsContainerConstIteratorType fixedPointEnd = fixedPoints->End();

    // MeshPointDataContainerConstIteratorType fixedPointDataIt =fixedNormals->Begin();
    // MeshPointDataContainerIteratorType mappedPointDataIt = mappedNormals->Begin();

    /* Transform all points and their normals by current transformation*/
    for (; fixedPointIt != fixedPointEnd; ++fixedPointIt, ++mappedPointIt) //,++fixedPointDataIt,++mappedPointDataIt)
    {
      const OutputPointType mappedPoint = this->m_Transform->TransformPoint(fixedPointIt->Value());
      mappedPointIt.Value() = mappedPoint;
      // this->TransformPointNormal(fixedPointIt->Value(),  fixedPointDataIt->Value(), mappedPointDataIt->Value()  );
    }
  } // end of loop over meshes

  // Since this is a dummy metric always return value = 0 and derivative = [0,...,0]

} // end GetValueAndDerivative()


/**
 * ******************* PrintSelf *******************
 */

template <class TFixedPointSet, class TMovingPointSet>
void
MeshPenalty<TFixedPointSet, TMovingPointSet>::PrintSelf(std::ostream & os, Indent indent) const
{
  Superclass::PrintSelf(os, indent);
  //
  //   if ( this->m_ComputeSquaredDistance )
  //   {
  //     os << indent << "m_ComputeSquaredDistance: True"<< std::endl;
  //   }
  //   else
  //   {
  //     os << indent << "m_ComputeSquaredDistance: False"<< std::endl;
  //   }
} // end PrintSelf()


} // end namespace itk

#endif // end #ifndef itkPolydataDummyPenalty_hxx

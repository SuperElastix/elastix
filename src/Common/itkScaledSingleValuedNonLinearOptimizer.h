#ifndef __ScaledSingleValuedNonLinearOptimizer_h
#define __ScaledSingleValuedNonLinearOptimizer_h

#include "itkSingleValuedNonLinearOptimizer.h"
#include "itkScaledSingleValuedCostFunction.h"

namespace itk
{
  /** \class ScaledSingleValuedNonLinearOptimizer
   *
   * Optimizers that inherit from this class optimize a scaled 
   * cost function F(y) instead of the original function f(x):
   *
   * y = x * s
   * F(y) = f(y/s)
   *
   * where y are the scaled parameters, x the original parameters
   * and s the scales.
   *
   * During optimization the inheriting classes should update the
   * ScaledCurrentPosition (y) instead of the CurrentPosition (y/s). 
   *  
   * When an optimizer needs the value at a (scaled) position y,
   * it should use the function this->GetScaledValue(y)
   * instead of the GetValue method. Similar for the derivative.
   *
   * Typically, in StartOptimization() the following line should be present:
   *   this->SetCurrentPosition(this->GetInitialPosition);
   * This makes sure that the initial position y_0 = x_0 * s, where x_0 is
   * the initial (unscaled) position entered by the user. 
   * 
   * Note that:
   * - GetScaledCurrentPosition returns the current y.
   * - GetCurrentPosition returns the current x = y/s. This array is only
   * computed when asked for by the user.
   * - It is NOT necessary to set the CurrentPosition!! In fact, it is 
   * not possible anymore: the SetCurrentPosition(param) method is overrided
   * and calls SetScaledCurrentPositon(param*scales).
   * - The ITK convention is to set the squared scales in the optimizer.
   * So, if you want a scaling s, you must call SetScales(s.*s) (where .* 
   * symbolises the element-wise product of s with s)
   *
   */

  class ScaledSingleValuedNonLinearOptimizer :
    public SingleValuedNonLinearOptimizer
  {
  public:

		/** Standard ITK-stuff. */
    typedef ScaledSingleValuedNonLinearOptimizer  Self;
    typedef SingleValuedNonLinearOptimizer        Superclass;
    typedef SmartPointer<Self>                    Pointer;
    typedef SmartPointer<const Self>              ConstPointer;

		/** Method for creation through the object factory. */
    itkNewMacro( Self );

		/** Run-time type information (and related methods). */
    itkTypeMacro( ScaledSingleValuedNonLinearOptimizer,
      SingleValuedNonLinearOptimizer );

		/** Typedefs inherited from the superclass. */
    typedef Superclass::MeasureType               MeasureType;
    typedef Superclass::ParametersType            ParametersType;
    typedef Superclass::DerivativeType            DerivativeType;
    typedef Superclass::CostFunctionType          CostFunctionType;

    typedef NonLinearOptimizer::ScalesType        ScalesType;
    typedef ScaledSingleValuedCostFunction        ScaledCostFunctionType;
    typedef ScaledCostFunctionType::Pointer       ScaledCostFunctionPointer;

    /** Configure the scaled cost function. This function 
     * sets the current scales in the ScaledCostFunction.
     * NB: it assumes that the scales entered by the user
     * are the squared scales (following the ITK convention).
     * Call this method in StartOptimization() and after
     * entering new scales.
		 */
    virtual void InitializeScales(void);

    /** Setting: SetCostFunction. */
    virtual void SetCostFunction(CostFunctionType *costFunction);

    /** Setting: Turn on/off the use of scales. Set this flag to false when no
     * scaling is desired.
		 */
    virtual void SetUseScales( bool arg );
    const bool GetUseScales(void) const;
    
    /** Get the current scaled position. */
    itkGetConstReferenceMacro( ScaledCurrentPosition, ParametersType );

    /** Get the current unscaled position: get the ScaledCurrentPosition
     * and divide each element through its scale.
		 */
    virtual const ParametersType & GetCurrentPosition() const;
    
    /** Get a pointer to the scaled cost function. */
    itkGetConstObjectMacro( ScaledCostFunction, ScaledCostFunctionType );

    /** Setting: set to 'true' if you want to maximize the cost function.
     * It forces the scaledCostFunction to negate the cost function value
      * and its derivative.
			*/
    itkBooleanMacro( Maximize );
    virtual void SetMaximize( bool _arg );
    itkGetConstMacro( Maximize, bool );

  protected:

		/** The constructor. */
    ScaledSingleValuedNonLinearOptimizer();
		/** The destructor. */
    virtual ~ScaledSingleValuedNonLinearOptimizer() {};

		/** PrintSelf. */
    void PrintSelf( std::ostream& os, Indent indent ) const{};
    
		/** Member variables. */
    ParametersType                  m_ScaledCurrentPosition;
    ScaledCostFunctionPointer       m_ScaledCostFunction;
    
    /** Set m_ScaledCurrentPosition. */ 
    virtual void SetScaledCurrentPosition( const ParametersType & parameters );

    /** Set the scaled current position by entering the non-scaled
     * parameters. The method multiplies param by the scales and
     * calls SetScaledCurrentPosition.
     *
     * Note: It is not possible (and needed) anymore to set m_CurrentPosition.
     * Optimizers that inherit from this class should optimize the scaled
     * parameters!
     *
     * This method will probably only be used to convert the InitialPosition
     * entered by the user.
     */
    virtual void  SetCurrentPosition ( const ParametersType &param );

    /** Divide the (scaled) parameters by the scales and call the GetValue routine 
     * of the unscaled cost function.
		 */
    virtual MeasureType GetScaledValue(
      const ParametersType & parameters ) const;

    /** Divide the (scaled) parameters by the scales, call the GetDerivative routine
     * of the unscaled cost function and divide the resulting derivative by
     * the scales.
		 */
    virtual void GetScaledDerivative(
      const ParametersType & parameters,
      DerivativeType & derivative ) const;

    /** Same procedure as in GetValue and GetDerivative. */
    virtual void GetScaledValueAndDerivative(
      const ParametersType & parameters,
      MeasureType & value,
      DerivativeType & derivative ) const;
        
  private:

		/** The private constructor. */
    ScaledSingleValuedNonLinearOptimizer( const Self& );	// purposely not implemented
		/** The private copy constructor. */
    void operator=( const Self& );												// purposely not implemented

    /** Variable to store the CurrentPosition, when the function
     * GetCurrentPosition is called. This method needs a member variable,
     * because the GetCurrentPosition return something by reference.
		 */
    mutable ParametersType          m_UnscaledCurrentPosition;

    bool                            m_Maximize;

  }; // end class ScaledSingleValuedNonLinearOptimizer


} // end namespace itk

#endif //#ifndef __ScaledSingleValuedNonLinearOptimizer_h

